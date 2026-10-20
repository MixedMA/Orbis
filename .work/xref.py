import sys, bisect, re, struct
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1]); import pefile
import numpy as np
pe = pefile.PE(sys.argv[2], fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
img = pe.get_memory_mapped_image(); base=pe.OPTIONAL_HEADER.ImageBase
funcs=sorted((r.struct.BeginAddress,r.struct.EndAddress) for r in pe.DIRECTORY_ENTRY_EXCEPTION); starts=[f[0] for f in funcs]
target=int(sys.argv[3],16)
text=[s for s in pe.sections if s.Name.startswith(b'.text')][0]
a=text.VirtualAddress; code=np.frombuffer(img[a:a+text.Misc_VirtualSize],dtype=np.uint8)
# find any 4-byte disp such that (pos+4+disp == target) for instruction end at pos+4 (approx: rip-rel disp is last 4 bytes, or followed by imm)
n=len(code)-4
d=np.frombuffer(code[:n//4*4].tobytes(),dtype='<i4')  # only aligned; do all offsets instead
hits=[]
c=code.astype(np.int64)
disp=(c[0:n]|(c[1:n+1]<<8)|(c[2:n+2]<<16)|(c[3:n+3]<<24)); disp=np.where(disp>=2**31,disp-2**32,disp)
pos=np.arange(n)+a
for extra in (0,1,4):
    m=np.nonzero(pos+4+extra+disp==target)[0]
    hits+= [int(pos[i]) for i in m]
import capstone
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
def cstr(r):
    s=img[r:r+100].split(b'\0')[0]
    return s.decode() if len(s)>=4 and all(32<=x<127 for x in s) else None
fs=sorted(set(funcs[bisect.bisect_right(starts,h)-1] for h in hits))
print(len(hits),'hits in',len(fs),'functions')
for b,e in fs[:60]:
    strs=[]
    for ins in md.disasm(img[b:e], base+b):
        mm=re.search(r'\[rip ([+-]) 0x([0-9a-f]+)\]', ins.op_str)
        if mm:
            t=ins.address+ins.size+int(mm.group(2),16)*(1 if mm.group(1)=='+' else -1)-base
            s=cstr(t)
            if s and 'gamestate' not in s and 'ThreadForbid' not in s and 'mnt' not in s: strs.append(s)
    if strs: print(hex(b), strs[:8])
