"""Summarize archived S/D/I/N counts; no new hypothesis/reference alignment."""
from pathlib import Path
import argparse,csv,json
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();totals={}
    with a.counts.open() as f:
        for line in f:
            r=json.loads(line)
            for method,sc in r['counts'].items():
                key=(method,r['lang']);v=totals.setdefault(key,dict(S=0,D=0,I=0,N=0,records=0))
                for k in ['S','D','I','N']:v[k]+=sc[k]
                v['records']+=1
    rows=[dict(strategy=m,lang=l,**v,WER=100*(v['S']+v['D']+v['I'])/v['N']) for (m,l),v in sorted(totals.items())]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Summarized archived counts without generating new predictions, edit alignments or intervals.')
if __name__=='__main__':main()
