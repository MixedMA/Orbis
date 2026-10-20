import sys, struct
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1])
import logging; logging.disable(logging.CRITICAL)
from minidump.minidumpfile import MinidumpFile
import pefile
mf = MinidumpFile.parse(sys.argv[2]); rd = mf.get_reader()
pe = pefile.PE(sys.argv[3], fast_load=True); img = pe.get_memory_mapped_image()
modbase = 0x7ff6e48b0000
def q(a):
    try: return struct.unpack('<Q', rd.read(a, 8))[0]
    except Exception: return None
def rtti(vt):
    col = q(vt - 8)
    if col is None: 
        # read from image instead
        r = vt - 8 - modbase; col = struct.unpack_from('<Q', img, r)[0]
    r = col - modbase
    td = struct.unpack_from('<i', img, r + 12)[0]
    return img[td + 16: td + 200].split(b'\0')[0].decode()
gs = 0x20b6bfb7bb0
for off in (0, 8, 0x10):
    o = q(gs + off)
    print('gamestate+%x ->' % off, hex(o) if o else o)
    if o:
        try:
            if modbase < o < modbase + 0x4000000: print('   vtable class', rtti(o))
            else:
                vt = q(o); print('   obj vtable', hex(vt) if vt else vt, rtti(vt) if vt else '')
        except Exception as e: print('   err', e)
# stack scan for values that look like small ids near rsp
rsp = 0x74c53fe6c0
vals = []
for i in range(0, 0x200, 4):
    try: vals.append(struct.unpack('<i', rd.read(rsp + i, 4))[0])
    except Exception: break
print('stack ints', vals[:64])
