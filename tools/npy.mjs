// Tiny .npy reader for tests (little-endian f4/u1/i4/i8, C order).
import fs from 'node:fs';
export function readNpy(path) {
  const b = fs.readFileSync(path), hl = b.readUInt16LE(8), h = b.toString('latin1', 10, 10 + hl);
  const dtype = /'descr': '([^']+)'/.exec(h)[1], shape = /'shape': \(([^)]*)\)/.exec(h)[1].split(',').filter(s => s.trim()).map(Number);
  const T = { '<f4': Float32Array, '|u1': Uint8Array, '<i4': Int32Array, '<i8': BigInt64Array }[dtype];
  const off = 10 + hl, ab = b.buffer.slice(b.byteOffset + off, b.byteOffset + b.length);
  return { data: new T(ab), shape };
}
