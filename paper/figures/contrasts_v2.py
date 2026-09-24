"""Redraw the twelve fixed contrasts; no new statistics or resampling."""
from vector_canvas import *
import hashlib

DATA=P/'contrasts_data.json'
SHORT={'Global':'Glo','Conf':'Conf','MWER-Global':'MWER-Glo','MWER-Conf':'MWER-Conf',
       'Text+Global':'T+G','Local':'Local','B5':'B5','Text':'Text'}

def family(row):
    return {'Original':'O','Fixed-checkpoint':'F','Exploratory':'M','Frozen-text':'T'}[row['group'].split()[0]]

def contrasts():
    rows=json.loads(DATA.read_text());by_name={r['name']:r for r in rows}
    s=SVG(504,232);rendered=[]
    for xx,yy,hh in [(0,0,129),(255,0,129),(0,132,76),(255,132,76)]:
        s.rect(xx+.5,yy+.5,248,hh-.5,'white','#C9D6E2',4,.55)
    s.rect(1,209,502,13,'#F8FAFC','#C9D6E2',3,.5)
    def panel(x,title,names,domain,ticks,ys,top,bottom):
        left,right=x+107,x+202
        plot_top=min(ys)-7
        xp=lambda v:left+(v-domain[0])/(domain[1]-domain[0])*(right-left)
        s.rect(x+1,top-11,247,18,'#F0F4F8','none',2)
        s.text(x+7,top+1,title,9.6,'bold')
        if top<20:s.text(x+244,top+19,'ΔWER',9,color=GRAY,anchor='end')
        for yy in ys:s.line([(x+5,yy+9),(x+247,yy+9)],'#E9EDF2',.35)
        for tick in ticks:
            tx=xp(tick)
            if tick!=0:s.line([(tx,plot_top),(tx,bottom)],'#E4EAF0',.4)
            s.line([(tx,bottom),(tx,bottom+2)],'#8698A8',.5)
            label=f'{tick:.2f}' if domain[1]-domain[0]<1 else f'{tick:g}'
            s.text(tx,bottom+12,label,9,color=GRAY,anchor='middle')
        s.line([(left,bottom),(right,bottom)],'#8698A8',.55)
        s.line([(xp(0),plot_top),(xp(0),bottom)],'#465966',.85,False,'2.2 2.2')
        for name,yy in zip(names,ys):
            r=by_name[name];f=family(r);co,_=FAMILY[f]
            assert domain[0]<=r['lo']<=r['mean']<=r['hi']<=domain[1]
            label=' − '.join(SHORT[v] for v in name.split(' − '))
            if name=='MWER-Global − MWER-Conf':label='MWER: Glo − Conf'
            if f == 'O': label += ' [101]'
            s.text(x+5,yy+3,label,9)
            low,mean,high=xp(r['lo']),xp(r['mean']),xp(r['hi'])
            s.line([(low,yy),(high,yy)],co,1.05)
            for xx in [low,high]:s.line([(xx,yy-2.3),(xx,yy+2.3)],co,.8)
            s.marker(mean,yy,f,co)
            s.text(x+245,yy+3,f"{r['mean']:+.4f}",9,anchor='end')
            rendered.append(dict(r,display_label=label,panel=title,family=f,
                                 axis_limits=list(domain),plot_x=[left,right],mean_x=mean,lo_x=low,hi_x=high,
                                 row_y=yy,zero_line_y=[plot_top,bottom]))
    panel(0,'(a) Shared-bank comparisons',
          ['Local − B5','Global − B5','MWER-Global − B5','Text − Global','Text+Global − Global'],
          (-2.3,.10),[-2,-1,0],[38,54,70,86,102],13,115)
    panel(255,'(b) Audio beyond confidence',
          ['Local − Conf','Global − Conf','MWER-Global − MWER-Conf'],
          (-.27,.015),[-.25,-.15,0],[45,72,99],13,115)
    panel(0,'(c) Alternative fitting scheme',
          ['MWER-Conf − Conf','MWER-Global − Global'],
          (-.27,.10),[-.2,-.1,0,.1],[163,181],143,192)
    panel(255,'(d) Near-zero increments · expanded axis',
          ['Local − Global','Text+Global − Text'],
          (-.035,.035),[-.03,0,.03],[163,181],143,192)
    for x,f,label in [(7,'O','Original 98.75%'),(126,'F','Checkpoints 98.33%'),
                      (275,'M','MWER 98.75%'),(409,'T','Text 98.33%')]:
        co,_=FAMILY[f];s.marker(x,214,f,co);s.text(x+7,217,label,9,color=GRAY)
    s.text(252,229,'CV27 ΔWER (pp); negative favors first. [101]: seed101; others: 3-checkpoint mean. Axes differ.',9,color=GRAY,anchor='middle')
    assert len(rows)==len(rendered)==len({r['name'] for r in rendered})==12
    result=s.save('contrasts');result.update(data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),statistics_recomputed=False,rendered_values=rendered)
    (P/'contrasts_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (P/'figure2_rendered_values.json').write_text(json.dumps(rendered,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':contrasts()
