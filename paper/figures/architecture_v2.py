"""Editable frozen-reranking architecture; no model or feature changes."""
from vector_canvas import *

def tensor(s,x,y,w,h,rgb,order):
    rows,cols=len(rgb),len(rgb[0])
    # Bin-edge ticks describe the same synthetic 80-bin array; no new signal.
    x += 10; w -= 10
    for label, yy in [('80',y+3),('40',y+h/2+3),('0',y+h+1)]:
        s.text(x-3,yy,label,9,color=GRAY,anchor='end')
    s.parts.append('<g shape-rendering="crispEdges">')
    for i in range(cols):
        for j in range(rows):
            co='#'+''.join(f'{int(v):02x}' for v in rgb[rows-1-j][i])
            s.rect(x+i*w/cols,y+j*h/rows,w/cols+.01,h/rows+.01,co,'none',0)
    s.parts.append('</g>')
    s.rect(x,y,w,h,'none','#455564',0,.5)
    colors=['#C7D8E8','#ECD6B4','#B9DCD0','#DACBE5','#D8DBAD','#E9BEBC','#C8D0D8','#B5DADF']
    for k,b in enumerate(order):
        s.rect(x+k*w/8,y+h+2,w/8-.4,10,colors[b],'none',0)
        s.text(x+(k+.5)*w/8-.2,y+h+10,str(b+1),9,color=INK,anchor='middle')

def stack(s,x,y,w,h):
    for d in [3,1.5,0]:s.rect(x+d,y-d,w,h,'#F5F8FC' if d else 'white','#9AAFC2',2,.65)

def arch():
    asset=json.loads((P/'synthetic_logmel_render.json').read_text())
    wave,clean,perm,order,mel=[asset[k] for k in ['waveform_display','clean_rgb','permuted_rgb','order','provenance']]
    s=SVG(504,276)
    s.rect(.5,.5,350,275,'white','#B9CADB',4,.65)
    s.rect(356,.5,147.5,275,'white','#B9CADB',4,.65)
    s.rect(.8,.8,349.4,20,'#EAF3FA','none',3)
    s.rect(356.3,.8,146.9,20,'#EAF3FA','none',3)
    s.text(3,12,'(a) Frozen models and shared candidate evidence',10,'bold')
    s.text(361,12,'(b) Alternative selectors',10,'bold')
    # Separate panels frame the frozen evidence path and alternative fits.
    ww=wave;s.line([(5+i/(len(ww)-1)*79,28+6*v) for i,v in enumerate(ww)],BLUE,.65)
    s.text(5,46,'Clean log-mel X',9,'bold',BLUE)
    tensor(s,5,52,80,39,clean,list(range(8)))
    s.line([(44,106),(44,123)],ORANGE,.9,True,edge='Fixed permutation of 20-frame blocks; no acoustic time alignment')
    s.text(51,117,'T',9,'bold',ORANGE)
    s.text(5,138,'Permuted T(X)',9,'bold',ORANGE)
    tensor(s,5,144,80,39,perm,order)
    s.text(5,209,'Mel-bin index',9,color=GRAY);s.text(5,221,'Synthetic, 1.6 s',9,color=GRAY)
    s.rect(104,24,236,30,'#F2F5F9','#B7C6D5',3,.6)
    s.text(113,36,'Shared bank: b, y₂, …, yₖ  (K ≤ 6)',9.2,'bold')
    s.text(113,48,'Clean greedy + beam-5 → deduplicate',9,color=GRAY)
    s.text(107,72,'Whisper-small + LoRA',9.2,'bold',BLUE);s.lock(239,62,BLUE)
    s.rect(104,78,146,110,'#F4F9FD','#8DB4D1',4,.8)
    s.text(116,91,'Encoder ×12',9,'bold');s.text(188,91,'Decoder ×12',9,'bold')
    stack(s,113,103,54,73);stack(s,188,103,54,73)
    s.rect(117,111,46,20,PB,BLUE,2,.6);s.text(140,124,'Self-attn',9,anchor='middle')
    s.rect(117,151,46,18,'#EFF2F6','#94A4B3',2,.6);s.text(140,163.5,'FFN',9,anchor='middle')
    s.line([(140,132),(140,148)],GRAY,.75,True)
    s.rect(192,109,46,23,PB,BLUE,2,.6);s.text(215,119,'Masked',9,anchor='middle');s.text(215,129,'self-attn',9,anchor='middle')
    s.rect(192,138,46,15,'#E6F1F2','#35888D',2,.6);s.text(215,148.5,'Cross-attn',9,anchor='middle')
    s.rect(192,161,46,12,'#EFF2F6','#94A4B3',2,.6);s.text(215,170.5,'FFN',9,anchor='middle')
    s.line([(215,133),(215,137)],GRAY,.6,True);s.line([(215,154),(215,160)],GRAY,.6,True)
    s.line([(85,72),(94,72),(94,119),(116,119)],BLUE,.9,True,edge='Clean log-mel enters the shared encoder')
    s.line([(85,164),(99,164),(99,127),(116,127)],ORANGE,.9,True,edge='Permuted log-mel enters the same encoder in a separate pass only for Global, Local and Text+Global')
    s.line([(164,162),(177,162),(177,145),(191,145)],GRAY,.8,True,edge='Encoder output supplies cross-attention in each scoring pass')
    s.text(172,137,'h',9,color=GRAY)
    s.line([(216,54),(216,59),(258,59),(258,99),(215,99),(215,108)],GRAY,.7,True,edge='Each bank candidate supplies its own teacher-forcing prefix')
    s.line([(243,167),(265,167),(265,54)],GRAY,.7,True,edge='Same frozen decoder acquires clean-audio candidates before scoring')
    s.text(112,201,'Permuted pass only for:',9,color=GRAY)
    s.text(112,213,'Global / Local / Text+Global',9,color=GRAY)
    s.text(276,77,'Token log p',9,'bold')
    for yy,co,pale,label in [(95,BLUE,PB,'clean'),(131,ORANGE,PO,'perm.')]:
        s.text(277,yy-5,label,9,color=co)
        for k in range(5):s.rect(276+k*8,yy,7,11,pale,co,.5,.45)
        s.line([(251,yy+6),(275,yy+6)],co,.8,True,edge='Body-token teacher-forced log probabilities for '+label)
    s.line([(315,100),(328,100)],BLUE,.8,True,edge='Clean scores enter C, including LCS-local clean confidence');s.badge(337,100,'C',BLUE,PB)
    s.text(329,115,'incl. LCS',9,color=GRAY,anchor='middle')
    s.line([(337,54),(337,92)],GRAY,.7,True,edge='Candidate-bank text, lengths and consensus also contribute to the eight base features C')
    s.line([(276,100),(270,100),(270,155),(288,155)],BLUE,.7,True)
    s.line([(296,143),(296,148)],ORANGE,.7,True)
    s.circle(296,156,6,'white',GRAY,.7);s.text(296,159.2,'−',10,'bold',anchor='middle')
    s.line([(296,163),(296,170),(278,170),(278,176)],ORANGE,.7,True,edge='All-token mean of clean-minus-permuted scores, relative to beam: g')
    s.line([(296,170),(331,170),(331,176)],PURPLE,.7,True,edge='Mean at text-LCS disagreement positions, relative to beam: l')
    s.badge(278,185,'g',ORANGE,PO);s.badge(331,185,'l',PURPLE,PP)
    s.text(278,205,'All tokens',9,color=GRAY,anchor='middle');s.text(326,217,'LCS tokens',9,color=GRAY,anchor='middle')
    s.line([(340,40),(346,40),(346,229),(183,229),(183,236)],GREEN,.7,True,edge='Same-bank candidate text enters frozen XLM-R; no references')
    s.rect(110,237,179,29,PG,GREEN,3,.65);s.lock(275,241,GREEN)
    s.text(118,249,'Frozen XLM-R-base',9.1,'bold',GREEN);s.text(118,261,'Mask each token → mean PLL',9)
    s.line([(290,252),(314,252)],GREEN,.8,True,edge='Candidate-minus-beam PLL supplies p');s.badge(327,252,'p',GREEN,PG)
    s.text(5,240,'Text LCS',9,'bold',PURPLE)
    for yy,lab,mid in [(253,'b','u₂'),(266,'y','v₂')]:
        s.text(5,yy,lab+':',9);s.text(20,yy,'u₁',9)
        s.rect(38,yy-9,17,12,PP,PURPLE,1,.45);s.text(46.5,yy,mid,9,anchor='middle');s.text(65,yy,'u₃',9)
    s.text(363,29,'Choose one configuration',9,color=GRAY)
    cols=[438,455,472,489]
    for xx,l,co in zip(cols,'Cglp',[BLUE,ORANGE,PURPLE,GREEN]):s.text(xx,46,l,9.2,'bold',co,'middle')
    s.line([(363,51),(499,51)],'#C3CFDB',.65)
    configs=[('Conf','C'),('Global','Cg'),('Local','Cgl'),('Text','Cp'),('Text+Global','Cgp')]
    for i,(name,keys) in enumerate(configs):
        yy=64+18*i
        if i%2==0:s.rect(361,yy-10,139,17,'#F5F7FA','none',0)
        s.text(365,yy+1,name,9)
        for xx,l,co in zip(cols,'Cglp',[BLUE,ORANGE,PURPLE,GREEN]):
            if l in keys:s.circle(xx,yy-2,2.4,co,co,.4)
            else:s.line([(xx-2,yy-2),(xx+2,yy-2)],'#C7D0D9',.6)
    s.text(363,153,'C: 8D; g, l, p: 1D each',9,color=GRAY)
    s.rect(365,166,133,39,'#FBF6EE','#BD955C',3,.65,'2 2')
    s.text(431.5,179,'TRAIN-only supervision',9,'bold',anchor='middle')
    s.text(431.5,189,'References → edit gains',9,anchor='middle');s.text(431.5,199,'FIT features → ridge fit',9,anchor='middle')
    s.line([(431,206),(431,222)],'#A98047',.8,True,'2 2',edge='Only locked TRAIN-fitted coefficients theta enter inference')
    s.text(439,215,'θ',9,color='#A98047')
    s.line([(362,143),(357,143),(357,239),(367,239)],GRAY,.8,True,edge='One feature configuration enters its separately fitted locked selector')
    s.rect(368,224,130,33,PB,BLUE,3,.7)
    s.text(433,237,'Locked linear selector',9.2,'bold',anchor='middle');s.text(433,250,'max gain > 0; else b',9,anchor='middle')
    s.line([(433,258),(433,264)],GRAY,.8,True,edge='Exactly one transcript is selected')
    s.text(433,272,'Selected transcript',9,'bold',anchor='middle')
    out=s.save('architecture');out.update(synthetic_spectrogram=mel,configuration_matrix={n:list(k) for n,k in configs},model_updates=False)
    (P/'architecture_inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out

if __name__=='__main__':arch()
