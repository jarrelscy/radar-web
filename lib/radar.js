// RADAR inference on a preprocessed volume, following evaluate() in
// inference_demo.py: sliding windows of (96, 256, 384) with 0.25 overlap, each
// organ scored once in the first window that holds it whole, then a crop centred
// on the organ for anything left over.
//
// The vision network runs as three ONNX parts so no full-resolution 3D tensor is
// ever held by the runtime (see tools/export.py):
//   part_a: stem + stage-2 first conv, on 8-slice slabs with a 1-slice halo
//   part_b: rest of the encoder + decoder to 1/4 resolution, and the three
//           token embeddings, on the whole window at 1/4 in-plane resolution
//   part_c: last decoder stage, seg head, x2 upsample and argmax, on slabs
// The attention head is small and runs in plain JS.
import { ORGANS, ORGAN_EN, TEST_ITEMS } from './labels.js';
import { ROI } from './preprocess.js';

const SLAB = 8, OVERLAP = 0.25, MARGIN = 2;
const TEST_ORGANS = new Set(TEST_ITEMS.map(k => k.split('_')[0]));
const en = name => ORGAN_EN[ORGANS.indexOf(name)] || name;

export async function loadRadar(ort, { load, device = 'wasm', log = () => {} }) {
  const t0 = performance.now();
  const ep = device === 'webgpu' ? ['webgpu'] : ['wasm'];
  const opts = { executionProviders: ep, graphOptimizationLevel: 'all' };
  const [a, b0, b1, bm, c, hb, hj] = await Promise.all(
    ['part_a.onnx', 'part_b.data0', 'part_b.data1', 'part_b.onnx', 'part_c.onnx', 'head.bin', 'head.json'].map(load));
  const sessA = await ort.InferenceSession.create(a, opts);
  const sessB = await ort.InferenceSession.create(bm, { ...opts, externalData: [{ path: 'part_b.data0', data: b0 }, { path: 'part_b.data1', data: b1 }] });
  const sessC = await ort.InferenceSession.create(c, opts);
  const meta = JSON.parse(new TextDecoder().decode(hj));
  const hbuf = hb.buffer.slice(hb.byteOffset, hb.byteOffset + hb.byteLength);
  const head = { temp: meta.temp, keys: meta.text_keys };
  for (const [k, v] of Object.entries(meta.tensors)) head[k] = new Float32Array(hbuf, v.offset, v.shape.reduce((x, y) => x * y, 1));
  log(`models ready on ${ep[0]} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  return { ort, sessA, sessB, sessC, head, device: ep[0] };
}

// ---- windows (monai dense_patch_slices, first axis slowest)
export function windowStarts(size) {
  const starts = size.map((s, d) => {
    const iv = ROI[d] === s ? ROI[d] : Math.max(1, Math.trunc(ROI[d] * (1 - OVERLAP)));
    const num = Math.ceil(s / iv); let k = 0;
    while (k < num && !(k * iv + ROI[d] >= s)) k++;
    const cnt = k < num ? k + 1 : 1, out = [];
    for (let i = 0; i < cnt; i++) { const st = i * iv; out.push(st - Math.max(st + ROI[d] - s, 0)); }
    return out;
  });
  const out = [];
  for (const z of starts[0]) for (const y of starts[1]) for (const x of starts[2]) out.push([z, y, x]);
  return out;
}

function extract(img, start, size) {
  const [D, H, W] = img.dims, [d, h, w] = size, out = new Float32Array(d * h * w);
  for (let z = 0; z < d; z++) for (let y = 0; y < h; y++) {
    const s = ((start[0] + z) * H + start[1] + y) * W + start[2];
    out.set(img.data.subarray(s, s + w), (z * h + y) * w);
  }
  return out;
}

// Run the vision network on one window [d][h][w].
async function vision(R, win, [d, h, w], progress) {
  const { ort } = R, hw = h * w, h4 = h / 4, w4 = w / 4, tch = 128 * h4 * w4;
  const t = new Float32Array(d * tch);
  const nslab = Math.ceil(d / SLAB); let step = 0; const steps = 2 * nslab + 1;
  for (let z = 0; z < d; z += SLAB) {
    const z1 = Math.min(z + SLAB, d), k2 = z1 - z + 2;
    const inp = new Float32Array(k2 * hw), edge = new Float32Array(k2);
    for (let i = 0; i < k2; i++) {
      const zz = z - 1 + i;
      edge[i] = zz >= 0 && zz < d ? 1 : 0;
      const zc = Math.min(Math.max(zz, 0), d - 1);
      inp.set(win.subarray(zc * hw, (zc + 1) * hw), i * hw);
    }
    const o = await R.sessA.run({ img: new ort.Tensor('float32', inp, [k2, 1, h, w]), edge: new ort.Tensor('float32', edge, [k2]) });
    t.set(await o.t.getData(true), z * tch);
    progress(++step / steps);
  }
  const ob = await R.sessB.run({ t: new ort.Tensor('float32', t, [d, 128, h4, w4]) });
  const x2 = await ob.x2.getData(true);
  const emb = [await ob.emb1.getData(true), await ob.emb2.getData(true), await ob.emb3.getData(true)];
  progress(++step / steps);
  const label = new Int32Array(d * hw);
  for (let z = 0; z < d; z += SLAB) {
    const z1 = Math.min(z + SLAB, d), k = z1 - z;
    const o = await R.sessC.run({
      img: new ort.Tensor('float32', win.subarray(z * hw, z1 * hw), [k, 1, h, w]),
      x2: new ort.Tensor('float32', x2.subarray(z * tch, z1 * tch), [k, 128, h4, w4]),
    });
    label.set(await o.label.getData(true), z * hw);
    progress(++step / steps);
  }
  return { label, emb };
}

// Organs fully inside the window (no voxel within MARGIN of any face), ascending id.
function intactOrgans(label, [d, h, w], skipOrgan) {
  const present = new Uint8Array(37), boundary = new Uint8Array(37);
  for (let z = 0, i = 0; z < d; z++) {
    const zb = z < MARGIN || z >= d - MARGIN;
    for (let y = 0; y < h; y++) {
      const yb = zb || y < MARGIN || y >= h - MARGIN;
      for (let x = 0; x < w; x++, i++) {
        const l = label[i]; if (!l) continue;
        present[l] = 1;
        if (yb || x < MARGIN || x >= w - MARGIN) boundary[l] = 1;
      }
    }
  }
  if (skipOrgan !== undefined) boundary[skipOrgan + 1] = 0;
  const out = [];
  for (let l = 1; l < 37; l++) if (present[l] && !boundary[l]) out.push(l - 1);
  return out;
}

// Tokens (in d,h,w order) at each embedding scale that overlap organ `o`.
const POOL = [[8, 32, 32], [4, 16, 16], [2, 8, 8]];
function organKeys(label, [d, h, w], emb, o) {
  const keys = [];
  for (let s = 0; s < 3; s++) {
    const [pd, ph, pw] = POOL[s], th = h / ph, tw = w / pw, flag = new Uint8Array((d / pd) * th * tw);
    for (let z = 0, i = 0; z < d; z++) for (let y = 0; y < h; y++) {
      const rb = (Math.floor(z / pd) * th + Math.floor(y / ph)) * tw;
      for (let x = 0; x < w; x++, i++) if (label[i] === o + 1) flag[rb + Math.floor(x / pw)] = 1;
    }
    for (let k = 0; k < flag.length; k++) if (flag[k]) keys.push(emb[s].subarray(k * 256, (k + 1) * 256));
  }
  return keys;
}

// nn.MultiheadAttention(256, 4) with one query, then vision_projs[o] and L2 norm.
function organFeature(H, o, keys) {
  const E = 256, hd = 64, Wi = H.in_proj_weight, bi = H.in_proj_bias;
  const q0 = H.query_tokens.subarray(o * E, (o + 1) * E);
  const mv = (W, row0, x, n = E) => { const y = new Float32Array(n); for (let r = 0; r < n; r++) { let s = 0; const b = (row0 + r) * E; for (let c = 0; c < E; c++) s += W[b + c] * x[c]; y[r] = s; } return y; };
  const q = mv(Wi, 0, q0); for (let r = 0; r < E; r++) q[r] += bi[r];
  const att = new Float32Array(E);
  for (let hh = 0; hh < 4; hh++) {
    // score_n = q_h . (Wk_h key_n + bk_h) / 8; the bias term is constant over n
    const kq = new Float32Array(E);
    for (let j = 0; j < hd; j++) { const qj = q[hh * hd + j], b = (E + hh * hd + j) * E; for (let c = 0; c < E; c++) kq[c] += qj * Wi[b + c]; }
    const sc = keys.map(k => { let s = 0; for (let c = 0; c < E; c++) s += kq[c] * k[c]; return s / 8; });
    let m = -Infinity; for (const v of sc) if (v > m) m = v; let z = 0; const p = sc.map(v => { const e = Math.exp(v - m); z += e; return e; });
    const agg = new Float32Array(E);
    keys.forEach((k, n) => { const pn = p[n] / z; for (let c = 0; c < E; c++) agg[c] += pn * k[c]; });
    const v = mv(Wi, 2 * E + hh * hd, agg, hd);
    for (let j = 0; j < hd; j++) att[hh * hd + j] = v[j] + bi[2 * E + hh * hd + j];
  }
  const y = mv(H.out_proj_weight, 0, att); for (let r = 0; r < E; r++) y[r] += H.out_proj_bias[r];
  const f = mv(H.vision_projs_weight.subarray(o * E * E, (o + 1) * E * E), 0, y);
  let nn = 0; for (let r = 0; r < E; r++) { f[r] += H.vision_projs_bias[o * E + r]; nn += f[r] * f[r]; }
  nn = Math.max(Math.sqrt(nn), 1e-12); for (let r = 0; r < E; r++) f[r] /= nn;
  return f;
}

function itemProb(H, feat, item) {
  const k = H.keys.indexOf(item), t = H.text_feat.subarray(k * 512, (k + 1) * 512);
  let l0 = 0, l1 = 0; for (let c = 0; c < 256; c++) { l0 += feat[c] * t[c]; l1 += feat[c] * t[256 + c]; }
  l0 /= H.temp; l1 /= H.temp;
  const m = Math.max(l0, l1); return Math.exp(l1 - m) / (Math.exp(l0 - m) + Math.exp(l1 - m));
}

// Score the intact, not yet scored test organs of one window.
async function scoreWindow(R, win, size, state, skipOrgan, progress) {
  const { label, emb } = await vision(R, win, size, progress);
  const scored = [];
  for (const o of intactOrgans(label, size, skipOrgan)) {
    const name = ORGANS[o];
    if (!TEST_ORGANS.has(name) || state.feat.has(name)) continue;
    const keys = organKeys(label, size, emb, o), feat = organFeature(R.head, o, keys);
    state.feat.set(name, feat); scored.push(name);
    for (const item of TEST_ITEMS) if (item.split('_')[0] === name) state.scores[item] = itemProb(R.head, feat, item);
  }
  return { label, scored };
}

export async function runRadar(R, img, { log = () => {}, progress = () => {} } = {}) {
  const t0 = performance.now(), size = img.dims;
  const starts = windowStarts(size);
  const state = { feat: new Map(), scores: {} }, trace = { windows: [], fallback: [] };
  // stitched segmentation for the fallback crops; later windows overwrite
  // earlier ones in the overlap (upstream averages probabilities there)
  const stitched = new Uint8Array(size[0] * size[1] * size[2]);
  log(`${starts.length} window(s) over ${size.join('x')}`);
  for (let wi = 0; wi < starts.length; wi++) {
    const st = starts[wi], tw = performance.now();
    const win = extract(img, st, ROI);
    const { label, scored } = await scoreWindow(R, win, ROI, state, undefined, p => progress((wi + p) / (starts.length + 1)));
    for (let z = 0; z < ROI[0]; z++) for (let y = 0; y < ROI[1]; y++) {
      const o = ((st[0] + z) * size[1] + st[1] + y) * size[2] + st[2], s = (z * ROI[1] + y) * ROI[2];
      for (let x = 0; x < ROI[2]; x++) stitched[o + x] = label[s + x];
    }
    trace.windows.push({ start: st, organs: scored });
    log(`window ${wi + 1}/${starts.length} at ${st.join(',')}: ${((performance.now() - tw) / 1000).toFixed(1)}s, scored ${scored.length ? scored.map(en).join(', ') : 'nothing new'}`);
  }
  // organs never whole inside a window: rerun on a crop centred on them
  const todo = TEST_ITEMS.filter(k => !(k in state.scores));
  for (const item of TEST_ITEMS) {
    if (item in state.scores) continue;
    const o = ORGANS.indexOf(item.split('_')[0]);
    const box = organBox(stitched, size, o + 1);
    if (!box) continue;
    const tw = performance.now();
    const { start, crop } = centerCrop(box, size);
    const padded = crop.map(c => Math.ceil(c / 32) * 32);
    const win = new Float32Array(padded[0] * padded[1] * padded[2]);
    const part = extract(img, start, crop);
    for (let z = 0; z < crop[0]; z++) for (let y = 0; y < crop[1]; y++)
      win.set(part.subarray((z * crop[1] + y) * crop[2], (z * crop[1] + y + 1) * crop[2]), (z * padded[1] + y) * padded[2]);
    const { scored } = await scoreWindow(R, win, padded, state, o, p => progress((starts.length + p * 0.99) / (starts.length + 1)));
    trace.fallback.push({ item, organ_id: o, box, start, crop_shape: padded, organs: scored });
    log(`crop around ${en(item.split('_')[0])} (${padded.join('x')} at ${start.join(',')}): ${((performance.now() - tw) / 1000).toFixed(1)}s, scored ${scored.length ? scored.map(en).join(', ') : 'nothing'}`);
  }
  const missing = TEST_ITEMS.filter(k => !(k in state.scores)).map(k => k.split('_')[0]);
  if (todo.length) log(`${todo.length} finding(s) needed a centred crop`);
  log(`inference ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  progress(1);
  return { scores: state.scores, organsNotFound: [...new Set(missing)], trace };
}

// masks_to_boxes_3d: [x_min, y_min, z_min, x_max, y_max, z_max], x = last axis
function organBox(lab, [d, h, w], l) {
  const b = [Infinity, Infinity, Infinity, -1, -1, -1];
  for (let z = 0, i = 0; z < d; z++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
    if (lab[i] !== l) continue;
    if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (z < b[2]) b[2] = z;
    if (x > b[3]) b[3] = x; if (y > b[4]) b[4] = y; if (z > b[5]) b[5] = z;
  }
  return b[3] < 0 ? null : b;
}

// center_crop in inference_demo.py, returning start (z,y,x) and size
function centerCrop(b, [d, h, w]) {
  const [x0, y0, z0, x1, y1, z1] = b;
  const cs = [Math.max(ROI[0], z1 - z0), Math.max(ROI[1], y1 - y0), Math.max(ROI[2], x1 - x0)];
  const c = [(z0 + z1) >> 1, (y0 + y1) >> 1, (x0 + x1) >> 1], lim = [d, h, w];
  const start = [0, 0, 0], crop = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    let s = Math.max(0, c[k] - (cs[k] >> 1)); const e = Math.min(lim[k], s + cs[k]);
    if (e - s < cs[k]) s = Math.max(0, e - cs[k]);
    start[k] = s; crop[k] = e - s;
  }
  return { start, crop };
}
