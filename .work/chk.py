import sys, re, glob
import numpy as np
from PIL import Image
mod = sys.argv[1]
img = np.array(Image.open(mod+'/map/provinces.bmp').convert('RGB')).astype(np.int64)
key = (img[...,0]<<16)|(img[...,1]<<8)|img[...,2]
col2id = {}; ptype={}; pterr={}
for l in open(mod+'/map/definition.csv'):
    p=l.strip().split(';')
    if len(p)<8: continue
    i=int(p[0]); col2id[(int(p[1])<<16)|(int(p[2])<<8)|int(p[3])]=i; ptype[i]=p[4]; pterr[i]=p[6]
u,inv=np.unique(key,return_inverse=True); inv=inv.reshape(key.shape)
lut=np.array([col2id.get(int(c),-1) for c in u]); pid=lut[inv]
print('unknown colours px:',(pid<0).sum())
adj=set()
for a,b in ((pid[:,:-1],pid[:,1:]),(pid[:-1,:],pid[1:,:])):
    m=a!=b
    for x,y in set(zip(a[m].tolist(),b[m].tolist())): adj.add((x,y)); adj.add((y,x))
# states
prov2state={}
for f in glob.glob(mod+'/history/states/*.txt'):
    t=open(f,encoding='utf-8-sig').read()
    sid=int(re.search(r'\bid\s*=\s*(\d+)',t).group(1))
    for p in re.search(r'provinces\s*=\s*\{([^}]*)\}',t).group(1).split():
        prov2state[int(p)]=sid
land=[i for i,t in ptype.items() if t=='land' and i>0]
print('land provs not in state:',[i for i in land if i not in prov2state][:20])
print('non-land in state:',[p for p in prov2state if ptype.get(p)!='land'][:20])
for l in open(mod+'/map/railways.txt'):
    v=list(map(int,l.split())); ps=v[2:]
    if v[1]!=len(ps): print('rail count mismatch',l)
    for a,b in zip(ps,ps[1:]):
        if (a,b) not in adj: print('rail non-adjacent',a,b)
        if ptype.get(a)!='land': print('rail on non-land',a)
for l in open(mod+'/map/supply_nodes.txt'):
    v=list(map(int,l.split()))
    if ptype.get(v[1])!='land' or v[1] not in prov2state: print('bad supply node',v)
seas={i for i,t in ptype.items() if t in('sea','lake')}
bad=0
for l in open(mod+'/map/buildings.txt'):
    p=l.strip().split(';')
    if p[1] in('naval_base_spawn','floating_harbor','naval_headquarters','naval_supply_hub','coastal_bunker','dockyard'):
        s=int(p[0]); sp=int(p[6])
        if sp==0: continue
        if ptype.get(sp)!='sea': bad+=1; 
        elif not any(prov2state.get(x)==s and (x,sp) in adj for x in prov2state): bad+=1
print('bad naval buildings',bad)
print('types',{t:list(ptype.values()).count(t) for t in set(ptype.values())})
sr={}
ids=[]
for f in glob.glob(mod+'/map/strategicregions/*.txt'):
    t=open(f,encoding='utf-8-sig').read()
    rid=int(re.search(r'\bid\s*=\s*(\d+)',t).group(1)); ids.append(rid)
    for p in re.search(r'provinces\s*=\s*\{([^}]*)\}',t).group(1).split():
        if int(p) in sr: print('prov in 2 regions',p)
        sr[int(p)]=rid
allp=[i for i in ptype if i>0]
print('regions',len(ids),'max',max(ids),'missing provs',[p for p in allp if p not in sr][:20], 'extra',[p for p in sr if p not in ptype][:10])
wp=[l for l in open(mod+'/map/weatherpositions.txt') if l.strip()]
print('weatherpositions lines',len(wp),'region ids',sorted(set(int(l.split(';')[0]) for l in wp))[:5], len(set(int(l.split(';')[0]) for l in wp)))
cont=[l.split(';')[7].strip() for l in open(mod+'/map/definition.csv')][1:]
print('continents',{c:cont.count(c) for c in set(cont)})
print(open(mod+'/map/continent.txt').read() if __import__('os').path.exists(mod+'/map/continent.txt') else 'no continent.txt in mod')
from collections import Counter, defaultdict
comp=defaultdict(Counter)
for p,r in sr.items(): comp[r][ptype[p]]+=1
mixed={r:dict(c) for r,c in comp.items() if len(c)>1}
print('mixed regions',mixed)
print('region type summary',Counter(tuple(sorted(c)) for c in comp.values()))
# sea connectivity
import collections
seaset={p for p,t in ptype.items() if t=='sea'}
nb=collections.defaultdict(set)
for a,b in adj: nb[a].add(b)
seen=set(); comps=[]
for s in seaset:
    if s in seen: continue
    st=[s]; seen.add(s); n=0
    while st:
        x=st.pop(); n+=1
        for y in nb[x]:
            if y in seaset and y not in seen: seen.add(y); st.append(y)
    comps.append(n)
print('sea components',sorted(comps,reverse=True)[:10], len(comps))
lk={p for p,t in ptype.items() if t=='lake'}
print('lakes coastal flag', Counter(l.split(';')[5] for l in open(mod+'/map/definition.csv') if ';lake;' in l))
print('sea provinces with land terrain', Counter(pterr[p] for p in seaset), Counter(pterr[p] for p in lk))
print('land terrains', Counter(pterr[p] for p in land))
# provinces split into multiple pieces
coast={int(l.split(';')[0]):l.split(';')[5] for l in open(mod+'/map/definition.csv')}
c1=[p for p in land if any(y in seaset for y in nb[p]) and coast[p]!='true']
c2=[p for p in land if not any(y in seaset for y in nb[p]) and coast[p]=='true']
print('coastal missing',len(c1),'coastal wrong',len(c2))
# small sea comps detail
seen=set()
for s in seaset:
    if s in seen: continue
    st=[s]; seen.add(s); grp=[]
    while st:
        x=st.pop(); grp.append(x)
        for y in nb[x]:
            if y in seaset and y not in seen: seen.add(y); st.append(y)
    if len(grp)<10: print('small sea comp',grp,'regions',{sr[g] for g in grp})
# regions connected?
for r in set(sr.values()):
    ps=[p for p in sr if sr[p]==r]; pset=set(ps); st=[ps[0]]; vis={ps[0]}
    while st:
        x=st.pop()
        for y in nb[x]:
            if y in pset and y not in vis: vis.add(y); st.append(y)
    if len(vis)!=len(ps): print('region',r,'disconnected',len(vis),'/',len(ps), comp[r])
# naval buildings in states with coastal
import os
for r in (1,48,49):
    ps=[p for p in sr if sr[p]==r]
    print('region',r, [(p, pterr[p], sorted(ptype[y][0]+str(y) for y in nb[p])[:6]) for p in ps[:5]])
    ys,xs=np.nonzero(np.isin(pid,ps)); print('  bbox',xs.min(),xs.max(),ys.min(),ys.max(),'px',len(ys))
print('isolated land provs (no land neighbour):',[p for p in land if not any(ptype.get(y)=='land' and y>0 for y in nb[p])][:40])
