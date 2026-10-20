import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
a=np.array(Image.open(sys.argv[1]).convert('RGB').resize((1000,1000),Image.NEAREST)).astype(int)
q=(a//24)*24+12
k=(q[...,0]<<16)|(q[...,1]<<8)|q[...,2]
mx=a.max(2); mn=a.min(2)
u,c=np.unique(k,return_counts=True)
res=[]
for v,n in zip(u,c):
    if n<400: continue
    col=np.array([v>>16,(v>>8)&255,v&255])
    if col.max()<45: continue                 # black / near black
    if abs(col[0]-col[1])<20 and abs(col[1]-col[2])<20 and col.max()>150: continue  # white/light grey labels
    if col[2]>col[0]+40 and col.max()<130 and col[0]<40: pass
    m=ndimage.binary_opening(k==v,iterations=2)
    lab,nl=ndimage.label(m)
    if nl==0: continue
    sz=ndimage.sum(m,lab,range(1,nl+1))
    for i,s in enumerate(sz,1):
        if s<150: continue
        ys,xs=np.nonzero(lab==i)
        res.append((tuple(col),int(s),int(xs.mean()),int(ys.mean())))
res.sort(key=lambda r:-r[1])
img=Image.fromarray(a.astype(np.uint8)).convert('RGB'); d=ImageDraw.Draw(img)
for j,(col,s,x,y) in enumerate(res,1):
    d.rectangle([x-12,y-9,x+12,y+9],fill=(255,255,0)); d.text((x-9,y-7),str(j),fill=(0,0,0))
    print(j,col,s,f'x={x/10:.0f}% y={y/10:.0f}%')
img.save(sys.argv[2])
