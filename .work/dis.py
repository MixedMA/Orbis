import sys, bisect, re
sys.path.insert(0, sys.argv[1])
import pefile, capstone
pe = pefile.PE(sys.argv[2], fast_load=True)
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'], pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
base = pe.OPTIONAL_HEADER.ImageBase
exp = {}
for e in pe.DIRECTORY_ENTRY_EXPORT.symbols:
    if e.name:
        n = e.name.decode(); exp[n.split('@')[0].lstrip('?')] = e.address
sw = exp['PHYSFS_swapULE64']; wr = exp['PHYSFS_writeULE64']
funcs = sorted((r.struct.BeginAddress, r.struct.EndAddress) for r in pe.DIRECTORY_ENTRY_EXCEPTION)
starts = [f[0] for f in funcs]
img = pe.get_memory_mapped_image()
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); md.detail = False
def cstr(rva):
    try:
        s = img[rva:rva+120]; s = s.split(b'\0')[0]
        if len(s) >= 4 and all(32 <= c < 127 for c in s): return s.decode()
    except Exception: pass
    return None
def strings_in(beg, end):
    out = []
    for ins in md.disasm(img[beg:end], base+beg):
        m = re.search(r'\[rip ([+-]) 0x([0-9a-f]+)\]', ins.op_str)
        if m:
            off = int(m.group(2), 16) * (1 if m.group(1) == '+' else -1)
            t = ins.address + ins.size + off - base
            s = cstr(t)
            if s: out.append(s)
    return out
for name, off in [(n, int(o)) for n, o in (a.split(':') for a in sys.argv[3:])]:
    rva = (sw if name == 's' else wr) + off
    i = bisect.bisect_right(starts, rva) - 1
    b, e = funcs[i]
    # walk back over chained unwind fragments: use the outer function start approx
    ss = strings_in(b, e)
    print(f'frame rva {rva:#x} func {b:#x}-{e:#x} size {e-b}')
    print('   ', ss[:25])
