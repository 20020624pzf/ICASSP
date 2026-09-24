"""Inference adapters for an already admitted worker; importing loads no model.

The caller must obtain a real site lease before constructing Engine or PLL.
Historical site launchers and authorization records are not distributed.
This module never claims a lease, starts a process or downloads a model.
"""
import hashlib
import json
from pathlib import Path
import os
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
# Resolve only our loader by file path; never prepend generic modules to sys.path.
import importlib.util as _import_util
_loader_spec = _import_util.spec_from_file_location("_asr_vendor_loader", HERE / "asr_vendor_loader.py")
_loader = _import_util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader)
load_vendor = _loader.load_vendor


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def load_whisper_in_admitted_worker(seed, device='cuda:0'):
    """Only call after the site's genuine admission/lease succeeds."""
    cfg=json.loads(Path(os.environ['ASR_REPRO_CONFIG']).read_text())
    versions=json.loads((HERE.parent/'config/VERSIONS_AND_ASSETS.json').read_text())
    expected=next(v for v in versions['Whisper']['adapters'] if v['seed']==seed)
    adapter=Path(cfg['adapter_directories'][str(seed)])
    assert adapter.name=='adapter', 'Archived Engine expects an adapter directory with this basename'
    assert file_sha(adapter/'adapter_model.safetensors')==expected['sha256']
    assert Path(cfg['whisper_snapshot']).name==versions['Whisper']['revision']
    audio = load_vendor("audio_core")
    if load_vendor("common").CFG != cfg:
        raise RuntimeError("ASR_REPRO_CONFIG changed after helper initialization; start a fresh admitted worker")
    return audio.Engine(seed,device=device)


def load_pll_in_admitted_worker(batch=32,device='cuda:0'):
    """Only call after real site admission; verify actual tokenizer/model files."""
    cfg=json.loads(Path(os.environ['ASR_REPRO_CONFIG']).read_text())
    versions=json.loads((HERE.parent/'config/VERSIONS_AND_ASSETS.json').read_text())
    root=Path(cfg['xlmr_snapshot'])
    for filename,item in versions['XLM_R']['files'].items():
        assert file_sha(root/filename)==item['sha256'], filename
    PLL = load_vendor("pll").PLL
    model=PLL(str(root),batch=batch,device=device)
    model.reproduction_identity_verified=file_sha(HERE.parent/'config/PLL_IDENTITY.json')
    return model


def records(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def generate(engine, manifest, mode, output):
    """mode G or N5; manifest contains only uid/lang/path, never references."""
    assert mode in ['G','N5']
    with Path(output).open('x') as f:
        for r in manifest:
            assert 'text' not in r and 'reference' not in r
            feat, mask, meta = engine.features(r['path'])
            f.write(json.dumps(dict(uid=r['uid'],lang=r['lang'],
                items=engine.generate(feat,mask,r['lang'],mode),audio_meta=meta),ensure_ascii=False)+'\n')


def acoustic_features(engine, manifest, greedy_path, beam_path, output):
    audio = load_vendor("audio_core")
    deduplicate, perturb, assemble_features = audio.deduplicate, audio.perturb, audio.assemble_features
    manifest = list(manifest)
    g = {(r['lang'],r['uid']):r for r in records(greedy_path)}
    b = {(r['lang'],r['uid']):r for r in records(beam_path)}
    assert set(g) == set(b) == {(r['lang'],r['uid']) for r in manifest}
    with Path(output).open('x') as f:
        for r in manifest:
            assert 'text' not in r and 'reference' not in r
            key = r['lang'],r['uid']
            cc, mapping = deduplicate(b[key]['items'],g[key]['items'],r['lang'])
            meta, permutation = None, None
            if len(cc)>1:
                feat,mask,meta = engine.features(r['path'])
                perm,permutation = perturb(feat,mask,r['uid'],r['lang'])
                clean_encoder,perm_encoder = engine.encode(feat),engine.encode(perm)
                for c in cc:
                    clean,body = engine.logps(feat,mask,r['lang'],c['token_ids'],clean_encoder)
                    changed,body2 = engine.logps(perm,mask,r['lang'],c['token_ids'],perm_encoder)
                    assert body == body2
                    c.update(clean_logp=clean,perturbed_logp=changed,body_mask=body)
            f.write(json.dumps(dict(uid=r['uid'],lang=r['lang'],candidates=cc,
                original_mapping=mapping,feature_data=assemble_features(cc,r['lang']),
                perturbation=permutation,audio_meta=meta),ensure_ascii=False)+'\n')


def score_pll(pll_model, feature_paths, output_db):
    """Exact-text reuse under the archived model/tokenizer/window/dtype identity."""
    identity = hashlib.sha256((HERE.parent/'config/PLL_IDENTITY.json').read_bytes()).hexdigest()
    assert getattr(pll_model,'reproduction_identity_verified',None)==identity
    db = sqlite3.connect(output_db)
    db.execute('create table if not exists metadata (name text primary key,value text)')
    db.execute('create table if not exists texts (key text primary key,pll real,ntokens integer,windowed integer)')
    old = db.execute('select value from metadata where name=?',('PLL_IDENTITY_SHA256',)).fetchone()
    if old:
        assert old[0] == identity
    else:
        assert db.execute('select count(*) from texts').fetchone()[0] == 0
        db.execute('insert into metadata values (?,?)',('PLL_IDENTITY_SHA256',identity))
    db.commit()
    for path in feature_paths:
        for row in records(path):
            for c in row['candidates']:
                key = hashlib.sha256(c['text'].encode()).hexdigest()
                if db.execute('select 1 from texts where key=?',(key,)).fetchone():
                    continue
                value = pll_model.score_text(c['text'])
                db.execute('insert into texts values (?,?,?,?)',
                    (key,value['pll'],value['ntokens'],int(value['windowed'])))
                db.commit()
    db.close()
