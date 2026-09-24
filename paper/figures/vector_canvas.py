"""Two editable publication figures. No inference, fitting or statistical recomputation."""
from pathlib import Path
from xml.sax.saxutils import escape
import json, math
import cairosvg
P=Path(__file__).resolve().parent
FONT='Liberation Sans, Arial, sans-serif'
INK,GRAY,LINE='#202B33','#56616B','#C8D1D8'
BLUE,ORANGE,PURPLE,GREEN='#0072B2','#B86B08','#92569D','#00866B'
PB,PO,PP,PG='#E5F0FA','#FFF0D5','#F0E6F3','#E1F2EC'
FAMILY={'O':('#C76B24','#F6E4D7'),'F':(BLUE,PB),'M':(PURPLE,PP),'T':(GREEN,PG)}
class SVG:
 def __init__(self,w,h):
  self.w,self.h=w,h; self.texts=[]; self.edges=[]
  self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
 def rect(self,x,y,w,h,fill='white',stroke=LINE,r=3,sw=.7,dash=None):
  ds=f' stroke-dasharray="{dash}"' if dash else ''
  self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{ds}/>')
 def text(self,x,y,t,size=9,weight='normal',color=INK,anchor='start'):
  size=max(9,size)  # Official minimum for principal figure labels.
  # Liberation Sans lacks letter subscripts k/t. Position ordinary glyphs as
  # SVG tspans rather than relying on unsupported Unicode characters.
  parts=[]; offset=False
  for char in t:
   if char in 'ₖₜ':
    sub={'ₖ':'K','ₜ':'t'}[char]
    parts.append(f'<tspan dy="{size*.24}" font-size="{size*.75}">{sub}</tspan>');offset=True
   elif offset:
    parts.append(f'<tspan dy="{-size*.24}" font-size="{size}">{escape(char)}</tspan>');offset=False
   else:parts.append(escape(char))
  self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{"".join(parts)}</text>')
  self.texts.append(dict(text=t,x=x,y=y,font_pt=size,anchor=anchor))
 def line(self,pts,color=GRAY,width=.8,arrow=False,dash=None,edge=None):
  d='M'+' L'.join(f'{x},{y}' for x,y in pts);ds=f' stroke-dasharray="{dash}"' if dash else ''
  self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{ds}/>')
  if arrow:
   a,b=pts[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);back=(b[0]-4*math.cos(ang),b[1]-4*math.sin(ang))
   p1=(back[0]+2.05*math.sin(ang),back[1]-2.05*math.cos(ang));p2=(back[0]-2.05*math.sin(ang),back[1]+2.05*math.cos(ang))
   self.parts.append(f'<path d="M{b[0]},{b[1]} L{p1[0]},{p1[1]} L{p2[0]},{p2[1]} Z" fill="{color}"/>')
  if edge:self.edges.append(dict(meaning=edge,points=pts,dashed=bool(dash)))
 def circle(self,x,y,r,fill='white',stroke=GRAY,sw=.8):
  self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
 def lock(self,x,y,color=BLUE):
  self.parts.append(f'<path d="M{x+1.7},{y+4} V{y+2.5} a2.3,2.3 0 0 1 4.6,0 V{y+4}" fill="none" stroke="{color}" stroke-width=".8"/>')
  self.rect(x,y+4,8,6,'white',color,1,.8);self.circle(x+4,y+6.8,.6,color,color,.3)
 def badge(self,x,y,t,color,fill):
  self.rect(x-6,y-7,13,13,fill,color,2,.65);self.text(x+.5,y+2.9,t,8.8,'bold',color,'middle')
 def marker(self,x,y,f,color):
  if f=='O':self.circle(x,y,2.35,'white',color,1.)
  elif f=='F':self.rect(x-2.2,y-2.2,4.4,4.4,color,color,0,.6)
  elif f=='M':self.parts.append(f'<path d="M{x},{y-2.8} L{x+2.7},{y+2.2} L{x-2.7},{y+2.2} Z" fill="{color}"/>')
  else:self.parts.append(f'<path d="M{x},{y-2.9} L{x+2.7},{y} L{x},{y+2.9} L{x-2.7},{y} Z" fill="{color}"/>')
 def save(self,name):
  f=P/(name+'.svg');f.write_text('\n'.join(self.parts+['</svg>'])+'\n')
  cairosvg.svg2pdf(url=str(f),write_to=str(P/(name+'.pdf')))
  cairosvg.svg2png(url=str(f),write_to=str(P/(name+'.png')),dpi=240)
  return dict(texts=self.texts,edges=self.edges,width_pt=self.w,height_pt=self.h)

