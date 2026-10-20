import sys, struct, bisect
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1]); import pefile
pe = pefile.PE(sys.argv[2], fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
base=pe.OPTIONAL_HEADER.ImageBase; img=pe.get_memory_mapped_image()
funcs=sorted((r.struct.BeginAddress,r.struct.EndAddress) for r in pe.DIRECTORY_ENTRY_EXCEPTION); starts=[f[0] for f in funcs]
def fo(r): i=bisect.bisect_right(starts,r)-1; return funcs[i][0]
targets={fo(int(a,16)):a for a in sys.argv[3:]}
lo,hi=min(targets)-0x4000,max(targets)+0x4000
rd=[s for s in pe.sections if s.Name.startswith(b'.rdata')][0]
a=rd.VirtualAddress; data=img[a:a+rd.Misc_VirtualSize]
import numpy as np
q=np.frombuffer(data[:len(data)//8*8],dtype='<u8')
cand=np.nonzero((q>=base+lo)&(q<base+hi))[0]
def tname(col):
    try:
        sig,off,cd,td,ch,selfr=struct.unpack_from('<IIIiii',img,col)
        name=img[td+16:td+200].split(b'\0')[0].decode()
        return name
    except Exception: return None
res={}
for k in cand:
    ptr=int(q[k])-base; f=fo(ptr)
    # walk back to vtable start: previous qword that is a COL pointer
    j=k
    while j>0:
        prev=int(q[j-1])-base
        if 0<prev<len(img):
            n=tname(prev)
            if n and n.startswith('.?A'):
                res.setdefault(f,set()).add((n, k-j)); break
        if int(q[j-1])<base or int(q[j-1])>base+len(img): break
        j-=1
        if k-j>400: break
for f,a in targets.items():
    print(f'func {f:#x} (frame {a}):', sorted(res.get(f,[]))[:5])
print('nearby:')
for f in sorted(res):
    print(f'{f:#x}', sorted(res[f])[:2])
