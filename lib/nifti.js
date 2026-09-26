// NIfTI-1 (.nii / .nii.gz) as a volume in the same form as series.js returns:
// dims, spacing, dir[a] = LPS direction of axis a, slice(k) -> Float32 along axis 2.

export async function niftiVolume(file, log = () => {}) {
  let buf = new Uint8Array(await file.arrayBuffer());
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    buf = new Uint8Array(await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  }
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getInt32(0, true) !== 348) throw new Error(dv.getInt32(0, false) === 348 ? 'big-endian NIfTI is not supported' : 'not a NIfTI-1 file');
  const ndim = dv.getInt16(40, true), dims = [1, 2, 3].map(i => dv.getInt16(40 + 2 * i, true));
  if (ndim < 3) throw new Error(`need a 3D volume, got ${ndim}D`);
  const dtype = dv.getInt16(70, true), pixdim = [1, 2, 3].map(i => dv.getFloat32(76 + 4 * i, true));
  const voxOffset = Math.round(dv.getFloat32(108, true)) || 352;
  let slope = dv.getFloat32(112, true), inter = dv.getFloat32(116, true);
  if (!slope || !isFinite(slope)) { slope = 1; inter = 0; }
  const qcode = dv.getInt16(252, true), scode = dv.getInt16(254, true);

  let m;   // 3x3 RAS matrix, columns = voxel axes scaled by spacing
  if (scode > 0) {
    m = [0, 1, 2].map(r => [0, 1, 2].map(c => dv.getFloat32(280 + 16 * r + 4 * c, true)));
  } else if (qcode > 0) {
    const b = dv.getFloat32(256, true), c = dv.getFloat32(260, true), d = dv.getFloat32(264, true);
    const a = Math.sqrt(Math.max(0, 1 - b * b - c * c - d * d)), qfac = dv.getFloat32(76, true) < 0 ? -1 : 1;
    const R = [[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
               [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
               [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b]];
    m = R.map(row => [row[0] * pixdim[0], row[1] * pixdim[1], row[2] * pixdim[2] * qfac]);
  } else {
    m = [[pixdim[0], 0, 0], [0, pixdim[1], 0], [0, 0, pixdim[2]]];   // no transform: treat as RAS-aligned
  }
  // RAS -> LPS, then split into spacing and unit directions per axis
  const spacing = [0, 1, 2].map(c => pixdim[c] || Math.hypot(m[0][c], m[1][c], m[2][c]) || 1);
  const dir = [0, 1, 2].map(c => {
    const v = [-m[0][c], -m[1][c], m[2][c]], n = Math.hypot(...v) || 1;
    return v.map(x => x / n);
  });

  const types = { 2: Uint8Array, 4: Int16Array, 8: Int32Array, 16: Float32Array, 64: Float64Array, 256: Int8Array, 512: Uint16Array, 768: Uint32Array };
  const T = types[dtype];
  if (!T) throw new Error(`NIfTI datatype ${dtype} not supported`);
  const plane = dims[0] * dims[1];
  const data = new T(buf.buffer.slice(buf.byteOffset + voxOffset, buf.byteOffset + voxOffset + plane * dims[2] * T.BYTES_PER_ELEMENT));
  log(`NIfTI ${dims.join('x')}, spacing ${spacing.map(s => s.toFixed(3)).join(', ')} mm, datatype ${dtype}`);
  return {
    dims, spacing, dir,
    async slice(k) {
      const out = new Float32Array(plane), s = data.subarray(k * plane, (k + 1) * plane);
      for (let i = 0; i < plane; i++) out[i] = s[i] * slope + inter;
      return out;
    },
  };
}
