// Volume -> RADAR input, matching radar-svc (DICOMOrient "LAS" + NIfTI) and
// DataFolder.__getitem__ in inference_demo.py:
//   resize to 1x1x5 mm (trilinear, align_corners=False, no antialiasing; the
//   in-plane target sizes use the swapped spacings exactly as upstream does),
//   clip to [-300, 400], min-max normalise, crop to the non-zero box with a
//   (5, 20, 20) margin, pad at the end to at least (96, 256, 384).
// Output is Float32 [D][H][W] with D = slices (towards S), H towards A, W towards L.
// Resizing is separable and commutes with flips, so the slice axis is resampled
// first and only the source slices that contribute are decoded.

export const ROI = [96, 256, 384];
const f32 = Math.fround;
const WANT = [1, -1, 1];   // LAS in LPS terms: +x, -y, +z

// PyTorch upsample (align_corners=False) source indices/weights for one axis.
function axisWeights(n, T) {
  const scale = f32(n / T), i0 = new Int32Array(T), i1 = new Int32Array(T), w1 = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let src = f32(f32(scale * (t + 0.5)) - 0.5);
    if (src < 0) src = 0;
    const a = Math.floor(src);
    i0[t] = a; i1[t] = a < n - 1 ? a + 1 : a; w1[t] = f32(src - a);
  }
  return { i0, i1, w1 };
}

export function orientation(vol) {
  // perm[a] = source axis that becomes LAS axis a; flip[a] if it runs the other way
  const perm = [-1, -1, -1], used = new Set();
  const order = [0, 1, 2].sort((p, q) => Math.max(...vol.dir[q].map(Math.abs)) - Math.max(...vol.dir[p].map(Math.abs)));
  for (const s of order) {
    let best = -1;
    for (let a = 0; a < 3; a++) if (perm[a] < 0 && (best < 0 || Math.abs(vol.dir[s][a]) > Math.abs(vol.dir[s][best]))) best = a;
    perm[best] = s; used.add(s);
  }
  const flip = perm.map((s, a) => Math.sign(vol.dir[s][a] || 1) !== WANT[a]);
  const n = perm.map(s => vol.dims[s]);
  // spacing as MONAI reads it back from the NIfTI affine: |diag| stored as float32
  const sp = perm.map((s, a) => f32(vol.spacing[s] * Math.abs(vol.dir[s][a])));
  return { perm, flip, n, sp };
}

export async function preprocess(vol, log = () => {}, progress = () => {}) {
  const t0 = performance.now();
  const { perm, flip, n, sp } = orientation(vol);
  const T = [Math.trunc(n[0] * sp[1]), Math.trunc(n[1] * sp[0]), Math.trunc(n[2] * (sp[2] / 5))].map(v => Math.max(1, v));
  log(`orient LAS: axes ${perm.join(',')} flips ${flip.map(Number).join(',')}; ${n.join('x')} @ ${sp.map(s => s.toFixed(3)).join(', ')} mm -> resize to ${T.join('x')}`);
  const inv = [0, 1, 2].map(s => perm.indexOf(s));
  const W = [0, 1, 2].map(a => axisWeights(n[a], T[a]));
  const src = (a, x) => flip[a] ? n[a] - 1 - x : x;   // LAS index -> source index along perm[a]

  // source axes 0/1 are the in-plane axes of each stored slice (0 fastest)
  const aS = inv[2], a0 = inv[0], a1 = inv[1];
  const n0 = vol.dims[0], n1 = vol.dims[1];
  const T0 = T[0], T1 = T[1], T2 = T[2], stride = [1, T0, T0 * T1];
  const out = new Float32Array(T0 * T1 * T2);

  // which source slices each output plane needs, and when each is last used
  const need = [], lastUse = new Map();
  for (let t = 0; t < T[aS]; t++) {
    const k0 = src(aS, W[aS].i0[t]), k1 = src(aS, W[aS].i1[t]);
    need.push([k0, k1]); lastUse.set(k0, t); lastUse.set(k1, t);
  }
  const uniq = [...new Set(need.flat())];
  log(`decoding ${uniq.length} of ${vol.dims[2]} source slices`);
  const cache = new Map(); let next = 0, done = 0;
  const get = k => { if (!cache.has(k)) cache.set(k, vol.slice(k).then(v => { progress(++done / uniq.length); return v; })); return cache.get(k); };
  const prefetch = () => { while (next < uniq.length && cache.size < 12) get(uniq[next++]); };

  // in-plane weights in source index order
  const wx = W[a0], wy = W[a1], Tx = T[a0], Ty = T[a1];
  const sx0 = new Int32Array(Tx), sx1 = new Int32Array(Tx), sy0 = new Int32Array(Ty), sy1 = new Int32Array(Ty);
  for (let t = 0; t < Tx; t++) { sx0[t] = src(a0, wx.i0[t]); sx1[t] = src(a0, wx.i1[t]); }
  for (let t = 0; t < Ty; t++) { sy0[t] = src(a1, wy.i0[t]); sy1[t] = src(a1, wy.i1[t]); }
  const plane = new Float32Array(n0 * n1), rowsX = new Float32Array(Tx * n1);
  for (let t = 0; t < T[aS]; t++) {
    prefetch();
    const [k0, k1] = need[t], w = W[aS].w1[t];
    const A = await get(k0), B = await get(k1);
    for (let i = 0; i < plane.length; i++) plane[i] = A[i] + (B[i] - A[i]) * w;
    for (let y = 0; y < n1; y++) {
      const r = y * n0;
      for (let x = 0; x < Tx; x++) { const u = plane[r + sx0[x]]; rowsX[y * Tx + x] = u + (plane[r + sx1[x]] - u) * wx.w1[x]; }
    }
    const base = t * stride[aS];
    for (let y = 0; y < Ty; y++) {
      const r0 = sy0[y] * Tx, r1 = sy1[y] * Tx, wv = wy.w1[y], ob = base + y * stride[a1];
      for (let x = 0; x < Tx; x++) { const u = rowsX[r0 + x]; out[ob + x * stride[a0]] = u + (rowsX[r1 + x] - u) * wv; }
    }
    for (const k of [k0, k1]) if (lastUse.get(k) === t) cache.delete(k);
  }
  log(`resized in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // clip, normalise
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < out.length; i++) {
    let v = out[i]; if (v > 400) v = 400; else if (v < -300) v = -300;
    out[i] = v; if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const den = f32(f32(mx - mn) + 1e-8);
  const lo = [T2, T1, T0], hi = [-1, -1, -1];
  for (let z = 0, i = 0; z < T2; z++) for (let y = 0; y < T1; y++) for (let x = 0; x < T0; x++, i++) {
    const v = f32(f32(out[i] - mn) / den); out[i] = v;
    if (v !== 0) {
      if (z < lo[0]) lo[0] = z; if (z > hi[0]) hi[0] = z;
      if (y < lo[1]) lo[1] = y; if (y > hi[1]) hi[1] = y;
      if (x < lo[2]) lo[2] = x; if (x > hi[2]) hi[2] = x;
    }
  }
  if (hi[0] < 0) throw new Error('volume is empty after windowing');
  // crop (end is max+extend, exclusive, as upstream) and pad at the end
  const size = [T2, T1, T0], ext = [5, 20, 20];
  const s0 = lo.map((v, d) => Math.max(v - ext[d], 0)), s1 = hi.map((v, d) => Math.min(v + ext[d], size[d]));
  const crop = s1.map((e, d) => e - s0[d]), dims = crop.map((c, d) => Math.max(c, ROI[d]));
  const img = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let z = 0; z < crop[0]; z++) for (let y = 0; y < crop[1]; y++) {
    const o = ((s0[0] + z) * T1 + s0[1] + y) * T0 + s0[2];
    img.set(out.subarray(o, o + crop[2]), (z * dims[1] + y) * dims[2]);
  }
  log(`window [-300, 400], crop ${crop.join('x')} from ${size.join('x')}, padded to ${dims.join('x')} (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  return { data: img, dims };
}
