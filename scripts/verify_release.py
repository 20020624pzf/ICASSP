"""Verify published files and already reported aggregate values; no model imports."""
from pathlib import Path
import ast
import csv
from decimal import Decimal
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(relative):
    return json.loads((ROOT / relative).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = read('MANIFEST.json')
    for row in manifest['files']:
        path = ROOT / row['path']
        assert path.is_file(), f"Missing: {row['path']}"
        assert path.stat().st_size == row['bytes'], row['path']
        assert sha(path) == row['sha256'], f"Hash mismatch: {row['path']}"

    for row in manifest['files']:
        if row['path'].endswith('.py'):
            path=ROOT/row['path']
            ast.parse(path.read_text(), filename=row['path'])

    all_models={}
    for filename in ['SELECTOR_MODELS.json', 'TEXT_MODELS.json']:
        all_models.update(read('reproduction/models/' + filename)['models'])
    for name, dim in [('Conf',8),('Global',9),('Local',10),('Text',9),('Text+Global',10)]:
        model=all_models[name]
        assert len(model['coef']) == len(model['scale']) == len(model['feature_names']) == dim
        assert all(float(x)>0 for x in model['scale'])

    with (ROOT/'reproduction/data/INTEGRATED_RESULTS.csv').open() as f:
        results=list(csv.DictReader(f))
    langs=['ar','fa','pl','tr','it']
    seeds=['101','202','303']
    def mean(split,method):
        values=[Decimal(r['WER']) for r in results if r['split']==split
                and r['strategy']==method and r['lang'] in langs and r['seed'] in seeds]
        assert len(values)==15, (split,method,len(values))
        return sum(values)/15
    comparisons=[]
    for split,first,second,expected in [
        ('TEST','Global','Conf','-0.1410'),('CV27','Global','Conf','-0.1824'),
        ('TEST','Text','Global','-0.9864'),('CV27','Text','Global','-1.2043'),
        ('CV27','Text+Global','Text','-0.0104'),
        ('CV27','Text+Global','Global','-1.2147'),('CV27','Local','Global','+0.0092')]:
        value=mean(split,first)-mean(split,second)
        assert f'{value:+.4f}'==expected, (split,first,second,str(value),expected)
        comparisons.append({'split':split,'contrast':first+' - '+second,'points':str(value)})
    freeze=read('reproduction/config/GENERATION_FREEZE.json')
    assert freeze['explicit_overrides']['max_new_tokens']==444
    assert freeze['explicit_overrides']['do_sample'] is False
    assert freeze['native_generation_config']['length_penalty']==1.0
    assert read('reproduction/config/PLL_IDENTITY.json')['max_body_tokens']==510
    with (ROOT/'reproduction/data/COST_TIMINGS_ANONYMOUS.csv').open() as f:
        cost=list(csv.DictReader(f))
    assert len(cost)==3000
    assert {int(r['recording_index']) for r in cost}==set(range(200))
    forbidden={'client_id','speaker_id','reference','transcript','audio_path','uid'}
    assert not forbidden.intersection(cost[0])
    tex=(ROOT/'paper/main.tex').read_text()
    assert 'ChatGPT' in tex and 'Codex' in tex
    bibliography=set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
    citations=set()
    for p in (ROOT/'paper').glob('*.tex'):
        for group in re.findall(r'\\cite\{([^}]+)\}',p.read_text()):citations.update(group.split(','))
    assert citations.issubset(bibliography)
    assert len(bibliography)==20
    print(json.dumps({'status':'PASS','hashed_files':len(manifest['files']),
        'fixed_selector_dimensions':[8,9,10,9,10],'reported_contrasts_checked':comparisons,
        'timing_rows':len(cost),'bibliography_entries':len(bibliography),
        'new_inference':False,'new_fitting':False,'new_statistical_tests':False},indent=2))

if __name__=='__main__':
    main()
