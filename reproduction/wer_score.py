"""Apply the unchanged WER-v3 script to author-supplied reference/hypothesis pairs."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from wer_rules_v3 import score
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True,help='JSONL: lang, reference, hypothesis; complete references')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();totals={}
    with a.pairs.open() as f:
        for line in f:
            r=json.loads(line);lang=r['lang'];assert lang in ['ar','fa','pl','tr','it']
            s=score(r['reference'],r['hypothesis'],lang)
            t=totals.setdefault(lang,dict(S=0,D=0,I=0,N=0,records=0))
            for k in ['S','D','I','N']:t[k]+=s[k]
            t['records']+=1
    assert set(totals)=={'ar','fa','pl','tr','it'},'Full five-language panel required'
    for t in totals.values():
        assert t['N']>0;t['WER']=100*(t['S']+t['D']+t['I'])/t['N']
    result={'per_language':totals,'macro_WER':sum(t['WER'] for t in totals.values())/5,
            'rule':'Unchanged WER-v3; no reference truncation or recording removal'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
