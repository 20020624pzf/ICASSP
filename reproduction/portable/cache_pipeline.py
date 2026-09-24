"""Path-independent TRAIN ridge fitting, locked selection and WER-v3 evaluation.

No inference, network access or model runtime is imported. Data stay local.
Input spec paths are resolved relative to the spec, never the author's machine.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import itertools
import json
import sqlite3
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
# Resolve only our loader by file path; never prepend generic modules to sys.path.
import importlib.util as _import_util
_loader_spec = _import_util.spec_from_file_location("_asr_vendor_loader", HERE / "asr_vendor_loader.py")
_loader = _import_util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader)
load_vendor = _loader.load_vendor
GainModel = load_vendor("risk_selector").GainModel
fit_gain_model = load_vendor("risk_selector").fit_gain_model
score = load_vendor("wer_rules_v3").score
words = load_vendor("wer_rules_v3").words

LANGS = ['ar', 'fa', 'pl', 'tr', 'it']
METHODS = ['Conf', 'Global', 'Local', 'Text', 'Text+Global']


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    with Path(p).open() as f:
        for line in f:
            yield json.loads(line)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def ref(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f'Refusing to replace {p}')
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def input_file(spec_path, item):
    p = (spec_path.parent/item['path']).resolve()
    assert sha(p) == item['sha256'], str(p)
    return p


def open_pll(spec_path, spec):
    p = (spec_path.parent/spec['pll_cache']).resolve()
    db = sqlite3.connect(p.as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    identity = db.execute('select value from metadata where name="PLL_IDENTITY_SHA256"').fetchone()
    assert identity and identity[0] == sha(HERE.parent/'config/PLL_IDENTITY.json')
    return db


def matrices(row, db):
    x = np.asarray(row['feature_data']['features'], dtype=np.float64)
    assert x.shape == (len(row['candidates']), 10)
    assert np.array_equal(x[0], np.zeros(10)) and np.isfinite(x).all()
    pll = []
    for c in row['candidates']:
        key = hashlib.sha256(c['text'].encode()).hexdigest()
        value = db.execute('select pll from texts where key=?', (key,)).fetchone()
        assert value and value[0] is not None, key
        pll.append(value[0])
    delta = np.asarray(pll)-pll[0]
    return {'Conf': x[:, :8], 'Global': x[:, :9], 'Local': x,
            'Text': np.column_stack([x[:, :8], delta]),
            'Text+Global': np.column_stack([x[:, :8], delta, x[:, 8]])}


def inherited_models():
    original = read(HERE.parent/'models/SELECTOR_MODELS.json')['models']
    text = read(HERE.parent/'models/TEXT_MODELS.json')['models']
    return {k: {**original, **text}[k] for k in METHODS}


def fit(args):
    sp = Path(args.spec).resolve()
    spec = read(sp)
    assert spec['split'] == 'TRAIN_CAL' and spec['seed'] == 101
    fp = input_file(sp, spec['features'])
    rp = input_file(sp, spec['references'])
    refs = {(r['lang'], r['uid']): r for r in rows(rp) if r['cal_split'] == 'FIT'}
    assert len(refs) == 1016
    den = {l: sum(len(words(r['text'], l)) for r in refs.values() if r['lang']==l) for l in LANGS}
    xs = {n: [] for n in METHODS}
    ys, ws, seen = [], [], set()
    with open_pll(sp, spec) as db:
        for r in rows(fp):
            key = (r['lang'], r['uid'])
            if key not in refs:
                continue
            assert key not in seen
            seen.add(key)
            n = len(r['candidates'])-1
            if not n:
                continue
            es = [sum(score(refs[key]['text'], c['text'], r['lang'])[k] for k in ['S','D','I']) for c in r['candidates']]
            ys.extend(es[0]-e for e in es[1:])
            ws.extend([1/(den[r['lang']]*n)]*n)
            for name, mat in matrices(r, db).items():
                xs[name].extend(mat[1:])
    assert seen == set(refs)
    inherited = inherited_models()
    models = {}
    for name in METHODS:
        m = fit_gain_model(xs[name], ys, ws, inherited[name]['feature_names'],
                           source_split='TRAIN_CAL_FIT', penalty=.01)
        models[name] = {'coef': m.coef.tolist(), 'scale': m.scale.tolist(),
                        'feature_names': list(m.feature_names)}
    save(args.out, {'models': models, 'fit_recordings': len(refs),
                    'feature_input': ref(fp), 'references': ref(rp),
                    'split': 'TRAIN_CAL_FIT', 'penalty': .01, 'intercept': False})


def select(args):
    sp = Path(args.spec).resolve()
    spec = read(sp)
    fp = input_file(sp, spec['features'])
    dd = read(args.models)['models'] if args.models else inherited_models()
    mm = {n: GainModel(np.asarray(dd[n]['scale']), np.asarray(dd[n]['coef']), tuple(dd[n]['feature_names'])) for n in METHODS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open_pll(sp, spec) as db, out.open('x') as f:
        for i, r in enumerate(rows(fp)):
            key = (r['lang'], r['uid'])
            assert key not in seen
            seen.add(key)
            xx = matrices(r, db)
            picks = {n: mm[n].choose(xx[n], 0) for n in METHODS}
            picks.update(B5=0, MBR=int(np.argmin(r['feature_data']['MBR'])),
                         **{'CD-seq': int(np.argmax(r['feature_data']['CD_seq']))})
            f.write(json.dumps({'row': i, 'lang': key[0], 'uid': key[1], 'selected': picks})+'\n')
    model_refs = [ref(args.models)] if args.models else [ref(HERE.parent/'models'/name) for name in ['SELECTOR_MODELS.json','TEXT_MODELS.json']]
    save(str(out)+'.lock.json', {'choices': ref(out), 'features': ref(fp),
                               'models': model_refs, 'reference_content_access': False,
                               'records': len(seen)})


def evaluate(args):
    sp = Path(args.spec).resolve()
    spec = read(sp)
    lk = read(str(args.choices)+'.lock.json')
    for v in [lk['choices'], lk['features'], *lk['models']]:
        assert sha(v['path']) == v['sha256']
    fp = input_file(sp, spec['features'])
    assert str(fp) == lk['features']['path']
    rp = input_file(sp, spec['references'])
    refs = {(r['lang'], r['uid']): r for r in rows(rp)}
    totals = collections.defaultdict(lambda: dict(S=0,D=0,I=0,N=0,recordings=0))
    seen = set()
    for r, selected in itertools.zip_longest(rows(fp), rows(args.choices)):
        assert r and selected
        key = (r['lang'], r['uid'])
        assert key == (selected['lang'], selected['uid']) and key not in seen
        seen.add(key)
        for name, index in selected['selected'].items():
            sc = score(refs[key]['text'], r['candidates'][index]['text'], r['lang'])
            t = totals[(name, r['lang'])]
            for k in ['S','D','I','N']:
                t[k] += sc[k]
            t['recordings'] += 1
    assert seen == set(refs)
    table = [dict(strategy=n,lang=l,**v,WER=100*(v['S']+v['D']+v['I'])/v['N']) for (n,l),v in totals.items()]
    save(args.out, {'per_language':table, 'reference_input':ref(rp),
                    'selection_lock':ref(str(args.choices)+'.lock.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['fit','select','evaluate'])
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', help='Optional fitted model file; defaults to original locked coefficients')
    parser.add_argument('--choices', help='Required for evaluation, with adjacent .lock.json')
    a = parser.parse_args()
    if a.phase == 'evaluate' and not a.choices:
        parser.error('--choices is required for evaluation')
    globals()[a.phase](a)


if __name__ == '__main__':
    main()
