"""Paired recording bootstrap of existing timing rows; no live timing or GPU."""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np

import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True)
a=p.parse_args()
Q=Path(__file__).resolve().parents[1]
OUT=Path(a.out)
OUT.mkdir(parents=True,exist_ok=False)
source=Q/'data/COST_TIMINGS_ANONYMOUS.csv'
systems = ['B5', 'Conf', 'Global', 'Text', 'Text+Global']
splits = ['TEST', 'CV27']
langs = ['ar', 'fa', 'pl', 'tr', 'it']
comparisons = [('Global', 'Conf'), ('Text+Global', 'Text'), ('Conf', 'B5')]
started = time.perf_counter()
with source.open() as f:
    raw = list(csv.DictReader(f))
assert len(raw) == 3000
records = {}
for r in raw:
    key = (r['physical_gpu'], r['partition'], r['recording_index'])
    cell = records.setdefault(key, {'split': r['split'], 'lang': r['lang'], 'quintile': r['quintile'], 'calls': {}})
    assert (cell['split'], cell['lang'], cell['quintile']) == (r['split'], r['lang'], r['quintile'])
    call = (r['system'], int(r['repetition']))
    assert call not in cell['calls']
    cell['calls'][call] = float(r['elapsed_seconds'])
assert len(records) == 200
strata = {}
point = np.zeros((2, 5))
for record in records.values():
    assert set(record['calls']) == {(s, repeat) for s in systems for repeat in range(3)}
    values = np.array([np.mean([record['calls'][(s, j)] for j in range(3)]) for s in systems])
    assert np.isfinite(values).all() and np.all(values > 0)
    key = (record['split'], record['lang'], record['quintile'])
    strata.setdefault(key, []).append(values)
    point[splits.index(record['split'])] += values
assert set(strata)=={(s,l,str(q)) for s in splits for l in langs for q in range(5)}
assert all(len(v)==4 for v in strata.values())
rng = np.random.default_rng(1729)
draws = np.zeros((10000, 2, 5))
for split in splits:
    for lang in langs:
        keys = sorted(k for k in strata if k[:2] == (split, lang))
        for key in keys:
            array = np.stack(strata[key])
            indices = rng.integers(0, len(array), size=(10000, len(array)))
            draws[:, splits.index(split), :] += array[indices].sum(axis=1)
table = []
for split in splits+['POOLED']:
    p = point.sum(axis=0) if split == 'POOLED' else point[splits.index(split)]
    d = draws.sum(axis=1) if split == 'POOLED' else draws[:, splits.index(split), :]
    for numerator, denominator in comparisons:
        i, j = systems.index(numerator), systems.index(denominator)
        rr = d[:, i]/d[:, j]
        lo, hi = np.quantile(rr, [.025, .975])
        table.append(dict(corpus=split, comparison=f'{numerator}/{denominator}', ratio_sum_mean_latency=p[i]/p[j], sampling_CI95_low=lo, sampling_CI95_high=hi, unique_recordings=200 if split=='POOLED' else 100, repeats_per_recording=3, draws=10000, interpretation='Descriptive sampling interval for fixed balanced recording mix and observed shared load; not simultaneous, not load-process uncertainty or deployment SLA'))
# Verify that collapsing repeats preserved the original ratio-of-sums estimand.
with (Q/'data/PAIRED_COST_COMPARISONS.csv').open() as f:
    old = list(csv.DictReader(f))
for r in table:
    if r['corpus'] != 'POOLED':
        numerator, denominator = r['comparison'].split('/')
        a = next(x for x in old if x['split']==r['corpus'] and x['first']==numerator and x['second']==denominator)
        assert abs(r['ratio_sum_mean_latency']-float(a['ratio_total_latency'])) < 1e-12
with (OUT/'COST_PAIRED_UNCERTAINTY.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
np.savez_compressed(OUT/'COST_BOOTSTRAP_DRAWS.npz', latency_sums=draws, systems=np.asarray(systems), splits=np.asarray(splits))
(OUT/'COST_UNCERTAINTY.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'raw_timing_rows':len(raw),'unique_recordings':len(records),'strata':50,'recordings_per_stratum':4,'original_point_ratios_verified':True,'source_GPU_contention_unchanged':True,'recording_repetitions_collapsed_before_resampling':True,'CPU_wall_seconds':time.perf_counter()-started,'results':table},indent=2)+'\n')
print(json.dumps({'status':'COST_SAMPLING_INTERVALS_COMPLETE','pooled':[x for x in table if x['corpus']=='POOLED']}),flush=True)
