import sys, numpy as np
from PIL import Image
from scipy import ndimage
from collections import Counter
im=Image.open(sys.argv[1]); print(im.mode, im.size)
a=np.array(im)
print('values',Counter(a.ravel().tolist()).most_common(20))
r=a<=11
lab,n=ndimage.label(r)  # 4-connected
print('rivers (4-conn comps)',n)
lab8,n8=ndimage.label(r,structure=np.ones((3,3)))
print('rivers (8-conn comps)',n8)
p=np.pad(r,1)
deg=(p[:-2,1:-1].astype(int)+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:])*r
print('degree hist',Counter(deg[r].tolist()))
bad=0
for i,sl in enumerate(ndimage.find_objects(lab),1):
    m=lab[sl]==i; s=(a[sl][m]==0).sum(); j=(a[sl][m]==1).sum()
    ends=((deg[sl]==1)&m).sum()
    if s!=1 or ends!=2 or j: bad+=1; print('comp',i,'size',m.sum(),'sources',s,'joins',j,'ends',ends,'at',sl[0].start,sl[1].start) if bad<15 else None
print('bad comps',bad)
# 2x2 river blocks
blk=r[:-1,:-1]&r[1:,:-1]&r[:-1,1:]&r[1:,1:]; print('2x2 blocks',blk.sum())
# sources: degree must be 1
print('sources not at end',((a==0)&(deg!=1)).sum())
