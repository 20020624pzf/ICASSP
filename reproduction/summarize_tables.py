"""Format previously evaluated CSV results; no predictions, fits or new intervals."""
from pathlib import Path
from decimal import Decimal
import argparse,csv,json,hashlib,re

P=Path(__file__).resolve().parent
LANGS=['ar','fa','pl','tr','it'];METHODS=['Conf','Global','Text','Text+Global']
def summarize(source,out):
    out.mkdir(parents=True,exist_ok=True)
    with source.open() as f:rows=list(csv.DictReader(f))
    keyed={}
    for r in rows:
        if r['split'] in ['TEST','CV27'] and r['lang'] in LANGS and r['strategy'] in METHODS+['Local']:
            k=(r['split'],r['seed'],r['lang'],r['strategy']);assert k not in keyed;keyed[k]=r
    assert len(keyed)==2*3*5*5
    mean=lambda split,lang,m:sum(Decimal(keyed[(split,str(s),lang,m)]['WER']) for s in [101,202,303])/3
    table=[]
    for split in ['TEST','CV27']:
        for lang in LANGS:
            v={m:mean(split,lang,m) for m in METHODS+['Local']}
            table.append(dict(corpus='FLEURS' if split=='TEST' else 'CV27',lang=lang,
                **{m:str(v[m]) for m in METHODS},
                **{'Global-Conf':str(v['Global']-v['Conf']),'Local-Global':str(v['Local']-v['Global']),
                   'Text-Global':str(v['Text']-v['Global']),'Text+Global-Text':str(v['Text+Global']-v['Text'])}))
    with (out/'LANGUAGE_ABSOLUTE_AND_DIFFERENCES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'LANGUAGE_ABSOLUTE_BY_CHECKPOINT.csv').open('w',newline='') as f:
        subset=[r for r in rows if r['split'] in ['TEST','CV27'] and r['lang'] in LANGS and r['strategy'] in METHODS]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(subset)
    tex=[r'\begin{table}[t]\centering\setlength{\tabcolsep}{2pt}',
         r'\caption{Language-level test WER (\%), averaged over three checkpoints. $\Delta_L=$Local--Global and $\Delta_T=$Text+Global--Text (points); positive is worse. Bold marks the lowest of the four absolute WERs, not significance. Glo/T+G denote Global/Text+Global.}\label{tab:languages}',
         r'\begin{tabular}{lrrrrrr}\toprule',
         r' & \multicolumn{4}{c}{Absolute WER} & \multicolumn{2}{c}{Differences} \\',
         r'Lang. & Conf & Glo & Text & T+G & $\Delta_L$ & $\Delta_T$ \\\midrule']
    for corpus in ['FLEURS','CV27']:
        if corpus=='CV27':tex.append(r'\midrule')
        tex.append(r'\multicolumn{7}{l}{\textit{'+corpus+r'}}\\')
        for r in [x for x in table if x['corpus']==corpus]:
            best=min(Decimal(r[m]) for m in METHODS)
            cells=[]
            for m in METHODS:
                s=f'{Decimal(r[m]):.3f}'
                cells.append(r'\textbf{'+s+'}' if Decimal(r[m])==best else s)
            cells += [f'{Decimal(r[k]):+.3f}' for k in ['Local-Global','Text+Global-Text']]
            tex.append(' & '.join([r['lang']]+cells)+r' \\')
    tex.append(r'\bottomrule\end{tabular}\end{table}')
    (out/'language_results.tex').write_text('\n'.join(tex)+'\n')
    # Keep every original language difference in a complete companion table.
    old=(P/'data/legacy_language_differences.tex').read_text()
    old_rows=re.findall(r'^(ar|fa|pl|tr|it) & (.*?) \\\\',old,re.M)
    assert len(old_rows)==10
    for (lang,cells),r in zip(old_rows,table):
        assert lang==r['lang']
        expected=[f'{Decimal(r[k]):+.3f}' for k in ['Global-Conf','Local-Global','Text-Global','Text+Global-Text']]
        assert [x.strip() for x in cells.split('&')]==expected,(lang,cells,expected)
    provenance={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'operation':'Arithmetic means across fixed checkpoints 101/202/303 of existing per-language WER; no re-scoring',
        'absolute_cells':40,'legacy_difference_cells_verified':40,'all_legacy_differences_retained':True,
        'original_intervals_recomputed':False}
    (out/'TABLE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Formatted 40 absolute WER cells; verified and retained all 40 original language differences.')

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--csv',type=Path,default=P/'data/INTEGRATED_RESULTS.csv');a.add_argument('--out',type=Path,required=True);args=a.parse_args();summarize(args.csv,args.out)
