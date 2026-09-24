"""Prepare local, reference-separated inputs from legally obtained original TSVs.

Configuration has split and languages: {lang: {tsv, tsv_sha256, audio_dir}}.
Paths are relative to the configuration. Does not download or extract datasets.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json

LANGS = ['ar','fa','pl','tr','it']


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prepare(config, out):
    cp = Path(config).resolve()
    cfg = json.loads(cp.read_text())
    split = cfg['split']
    assert split in ['TRAIN_CAL','DEV','TEST','CV27']
    assert set(cfg['languages']) == set(LANGS)
    manifest, refs = [], []
    for lang in LANGS:
        c = cfg['languages'][lang]
        p = (cp.parent/c['tsv']).resolve()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c['tsv_sha256']
        audio_dir = (cp.parent/c['audio_dir']).resolve()
        rr = []
        with p.open(encoding='utf-8') as f:
            if split == 'CV27':
                for r in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
                    client = r['client_id'].strip()
                    assert client
                    rr.append(dict(lang=lang,uid=Path(r['path']).stem,text=r['sentence'],
                        path=str(audio_dir/r['path']),client_cluster=h('F7_PRIVATE_CLIENT_V1|'+lang+'|'+client)))
            else:
                for r in csv.reader(f,delimiter='\t'):
                    if len(r)<3:
                        continue
                    rr.append(dict(lang=lang,uid=Path(r[1]).stem,text=r[2],path=str(audio_dir/r[1]),
                                   group=h(' '.join(r[2].split()).strip())[:16]))
        rr.sort(key=lambda r:r['uid'])
        assert len({r['uid'] for r in rr}) == len(rr)
        assert all(Path(r['path']).is_file() for r in rr), 'Missing audio; do not silently drop records'
        if split == 'TRAIN_CAL':
            groups = collections.defaultdict(list)
            for r in rr:
                groups[r['group']].append(r)
            selected = []
            for group in sorted(groups,key=lambda g:h(f'F7_CAL_V1|{lang}|{g}')):
                subset = 'FIT' if int(h(f'F7_FIT_CHECK_V1|{lang}|{group}'),16)/2**256 < .8 else 'CHECK'
                selected.extend(dict(r,cal_split=subset) for r in groups[group])
                if len(selected)>=256:
                    break
            rr = selected
        for r in rr:
            manifest.append({k:r[k] for k in ['uid','lang','path']})
            refs.append({k:v for k,v in r.items() if k!='path'})
    key = lambda r:(LANGS.index(r['lang']),r['uid'])
    manifest.sort(key=key); refs.sort(key=key)
    assert len(manifest)=={'TRAIN_CAL':1281,'DEV':1731,'TEST':3665,'CV27':58514}[split]
    if split=='TRAIN_CAL':
        assert sum(r['cal_split']=='FIT' for r in refs)==1016
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    (out/f'{split}_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False)+'\n')
    with (out/f'{split}_REFERENCES.jsonl').open('w') as f:
        for r in refs:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(a.config,a.out)
