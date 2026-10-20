import sys, bisect, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1]); import pefile, capstone
pe = pefile.PE(sys.argv[2], fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
img = pe.get_memory_mapped_image(); base=pe.OPTIONAL_HEADER.ImageBase
funcs=sorted((r.struct.BeginAddress,r.struct.EndAddress) for r in pe.DIRECTORY_ENTRY_EXCEPTION); starts=[f[0] for f in funcs]
md=capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
def cstr(r):
    s=img[r:r+120].split(b'\0')[0]
    return s.decode() if len(s)>=5 and all(32<=x<127 for x in s) else None
text=[s for s in pe.sections if s.Name.startswith(b'.text')][0]
a=text.VirtualAddress; code=img[a:a+text.Misc_VirtualSize]
# mov r64,[r64+0xb8] followed within 8 bytes by mov r32,[r64+0xc8]
pat=re.compile(rb'[\x48\x49\x4c\x4d]\x8b[\x80-\xbf]\xb8\x00\x00\x00(?:.{0,8}?)[\x40-\x4f]?\x8b[\x80-\xbf]\xc8\x00\x00\x00', re.S)
hits=[a+m.start() for m in pat.finditer(code)]
print(len(hits),'hits')
out={}
for h in hits:
    b,e=funcs[bisect.bisect_right(starts,h)-1]
    if b in out: continue
    strs=[]
    for ins in md.disasm(img[b:e], base+b):
        mm=re.search(r'\[rip ([+-]) 0x([0-9a-f]+)\]', ins.op_str)
        if mm:
            t=ins.address+ins.size+int(mm.group(2),16)*(1 if mm.group(1)=='+' else -1)-base
            s=cstr(t)
            if s and 'gamestate' not in s and 'ThreadForbid' not in s and not s.startswith('C:'): strs.append(s)
    out[b]=strs
for b,s in out.items():
    if s: print(hex(b), s[:10])
