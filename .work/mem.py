import sys, struct
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1])
import logging; logging.disable(logging.CRITICAL)
from minidump.minidumpfile import MinidumpFile
mf = MinidumpFile.parse(sys.argv[2]); rd = mf.get_reader()
def q(a):
    try: return struct.unpack('<Q', rd.read(a, 8))[0]
    except Exception as e: return None
def d(a):
    try: return struct.unpack('<i', rd.read(a, 4))[0]
    except Exception: return None
print('memory segments', len(mf.memory_segments_64.memory_segments) if mf.memory_segments_64 else len(mf.memory_segments.memory_segments))
r14=int(sys.argv[3],16); r15=int(sys.argv[4],16); r13=int(sys.argv[5],16)
print('area list ptr', hex(q(r14+0xb0) or 0), 'count', d(r14+0xbc))
for p in range(r15 - 0x40, r13 + 8, 8):
    e = q(p)
    if e is None: print(hex(p), 'unreadable'); continue
    x = q(e+0xb8) if e else None
    i = d(x+0xc8) if x else None
    print(hex(p), 'elem', hex(e), '+b8', hex(x) if x else x, 'id@c8', i, ' elem ints', [d(e+o) for o in range(0,0x40,4)] if e else None)
