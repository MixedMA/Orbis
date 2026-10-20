import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, sys.argv[1])
from minidump.minidumpfile import MinidumpFile
import pefile, capstone
mf = MinidumpFile.parse(sys.argv[2])
exc = mf.exception.exception_records[0]
er = exc.ExceptionRecord
print('code', er.ExceptionCode, 'addr', hex(er.ExceptionAddress), 'info', [hex(x) for x in er.ExceptionInformation[:er.NumberParameters]])
mod = [m for m in mf.modules.modules if m.name.lower().endswith('hoi4.exe')][0]
base = mod.baseaddress; print('hoi4 base', hex(base), 'rva', hex(er.ExceptionAddress - base))
tid = exc.ThreadId
th = [t for t in mf.threads.threads if t.ThreadId == tid][0]
rd = mf.get_reader()
import struct
ctx_loc = exc.ThreadContext
buf = rd.file_handle if hasattr(rd,'file_handle') else None
f = open(sys.argv[2], 'rb'); f.seek(ctx_loc.Rva); raw = f.read(ctx_loc.DataSize)
# CONTEXT x64: offsets
names = ['Rax','Rcx','Rdx','Rbx','Rsp','Rbp','Rsi','Rdi','R8','R9','R10','R11','R12','R13','R14','R15','Rip']
vals = struct.unpack_from('<17Q', raw, 0x78)
regs = dict(zip(names, vals))
for n in names: print(f'{n}={regs[n]:#x}', end='  ')
print()
pe = pefile.PE(sys.argv[3], fast_load=True); img = pe.get_memory_mapped_image()
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
rva = er.ExceptionAddress - base
for ins in md.disasm(img[rva-0x60:rva+0x20], pe.OPTIONAL_HEADER.ImageBase + rva - 0x60):
    mark = '>>' if ins.address - pe.OPTIONAL_HEADER.ImageBase == rva else '  '
    print(mark, hex(ins.address - pe.OPTIONAL_HEADER.ImageBase), ins.mnemonic, ins.op_str)
