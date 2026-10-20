import sys, struct
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1])
import logging; logging.disable(logging.CRITICAL)
from minidump.minidumpfile import MinidumpFile
import pefile
mf = MinidumpFile.parse(sys.argv[2]); rd = mf.get_reader()
pe = pefile.PE(r"F:/SteamLibrary/steamapps/common/Hearts of Iron IV/hoi4.exe", fast_load=True); img = pe.get_memory_mapped_image(); ib = pe.OPTIONAL_HEADER.ImageBase
modbase = 0x7ff6e48b0000
def q(a):
    try: return struct.unpack('<Q', rd.read(a, 8))[0]
    except Exception: return None
def d(a):
    try: return struct.unpack('<i', rd.read(a, 4))[0]
    except Exception: return None
def cls(obj):
    vt = q(obj)
    if not vt or not (modbase < vt < modbase + 0x4000000): return None
    try:
        col = struct.unpack_from('<Q', img, vt - modbase - 8)[0] - ib
        td = struct.unpack_from('<i', img, col + 12)[0]
        return img[td + 16:td + 120].split(b'\0')[0].decode()
    except Exception: return '?'
r14, r15, r13 = (int(x, 16) for x in sys.argv[3:6])
print('area class', cls(r14))
lst = q(r14 + 0xb0); n = d(r14 + 0xbc); print('count', n, 'crash idx', (r15 - lst) // 8)
for i in range(n):
    e = q(lst + 8 * i); x = q(e + 0xb8); pid = d(x + 0xc8) if x else None
    print(i, 'elem', cls(e), 'X', cls(x), 'id@X+c8', pid, ' elem ints 0x00-0x40:', [d(e + o) for o in range(8, 0x40, 4)])
print('--- searching id field')
elems=[q(lst+8*i) for i in range(n)]
cand={}
for off in range(0, 0x400, 4):
    vals=[d(e+off) for e in elems]
    if all(v is not None and 1<=v<=3739 for v in vals) and len(set(vals))==len(vals):
        cand[off]=vals
for off,vals in list(cand.items())[:6]: print(hex(off), vals)
