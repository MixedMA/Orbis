import sys, bisect, re
import sys as _s; _s.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1])
import pefile, capstone
pe = pefile.PE(sys.argv[2], fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
base = pe.OPTIONAL_HEADER.ImageBase
funcs = sorted((r.struct.BeginAddress, r.struct.EndAddress) for r in pe.DIRECTORY_ENTRY_EXCEPTION)
starts=[f[0] for f in funcs]
img = pe.get_memory_mapped_image()
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
def cstr(rva):
    s = img[rva:rva+100].split(b'\0')[0]
    if len(s) >= 4 and all(32 <= c < 127 for c in s): return s.decode()
    w = img[rva:rva+200]
    try:
        t = w.decode('utf-16le').split('\0')[0]
        if len(t)>=4 and t.isascii() and t.isprintable(): return 'W:'+t
    except Exception: pass
def func_of(rva):
    i=bisect.bisect_right(starts,rva)-1; return funcs[i]
def info(beg,end):
    strs=[];calls=[]
    for ins in md.disasm(img[beg:end], base+beg):
        m = re.search(r'\[rip ([+-]) 0x([0-9a-f]+)\]', ins.op_str)
        if m:
            t = ins.address+ins.size+int(m.group(2),16)*(1 if m.group(1)=='+' else -1)-base
            s=cstr(t)
            if s and 'gamestate.h' not in s and 'gamestate unit' not in s and 'ThreadForbid' not in s: strs.append(s)
        if ins.mnemonic=='call' and ins.op_str.startswith('0x'):
            calls.append(int(ins.op_str,16)-base)
    return strs,calls
for rva in [int(a,16) for a in sys.argv[3:]]:
    b,e=func_of(rva)
    strs,calls=info(b,e)
    print(f'== func {b:#x} crash@{rva:#x}: own strings {strs[:10]}')
    seen=set()
    for c in calls:
        cb,ce=func_of(c)
        if cb in seen: continue
        seen.add(cb)
        s2,_=info(cb,ce)
        if s2: print(f'   call {c:#x}: {s2[:6]}')
