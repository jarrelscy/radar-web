// Compare the ONNX segmentation with PyTorch, window by window (Dice per organ).
// Needs /tmp/radar_ref/{image,win0,win1}.npy from tools/seg-ref.py.
//   node tools/seg-test.mjs [input.zip|.nii.gz]
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { openInput } from '../lib/source.js';
import { pickSeries, seriesVolume } from '../lib/series.js';
import { niftiVolume } from '../lib/nifti.js';
import { preprocess } from '../lib/preprocess.js';
import { loadRadar, windowStarts, extract, vision } from '../lib/radar.js';
import { ORGAN_EN } from '../lib/labels.js';
import { readNpy } from './npy.mjs';

const require = createRequire(new URL('./package.json', import.meta.url));
const ort = require('onnxruntime-web');
ort.env.wasm.numThreads = 16;
const [input = '/tmp/radar_dcm/demo.zip'] = process.argv.slice(2);
const R = await loadRadar(ort, { load: async n => new Uint8Array(fs.readFileSync(new URL(`../models/${n}`, import.meta.url))), device: 'wasm' });
const REF = '/tmp/radar_ref/', ROI = [96, 256, 384];

function compare(name, a, b) {
  const inter = new Float64Array(37), na = new Float64Array(37), nb = new Float64Array(37); let same = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i], y = b[i]; na[x]++; nb[y]++; if (x === y) { inter[x]++; same++; } }
  const rows = [];
  for (let l = 1; l < 37; l++) if (na[l] + nb[l]) rows.push([ORGAN_EN[l - 1], 2 * inter[l] / (na[l] + nb[l]), na[l], nb[l]]);
  rows.sort((p, q) => p[1] - q[1]);
  console.log(`${name}: ${a.length - same} of ${a.length} voxels differ, ${rows.length} organs, lowest Dice:`);
  for (const [o, d, x, y] of rows.slice(0, 2)) console.log(`   ${o.padEnd(22)} Dice ${d.toFixed(4)}  voxels js ${x} / torch ${y}`);
  console.log(`   median Dice ${rows[rows.length >> 1][1].toFixed(4)}; organs present in only one: ${rows.filter(r => !r[2] || !r[3]).map(r => r[0]).join(', ') || 'none'}`);
}

const refImg = readNpy(REF + 'image.npy'), size = refImg.shape.slice(-3);
const vol = await (async () => { const f = new File([await fs.openAsBlob(input)], input.split('/').pop()); const inp = await openInput([f], () => {}); return inp.kind === 'nifti' ? niftiVolume(inp.file, () => {}) : seriesVolume((await pickSeries(inp.entries, () => {})).best.files, () => {}); })();
const jsImg = await preprocess(vol, () => {});
let md = 0; for (let i = 0; i < refImg.data.length; i++) md = Math.max(md, Math.abs(refImg.data[i] - jsImg.data[i]));
console.log(`preprocessed: js ${jsImg.dims.join('x')} vs torch ${size.join('x')}, max |diff| ${md.toExponential(2)}`);
const starts = windowStarts(size);
for (let wi = 0; wi < starts.length; wi++) {
  const want = readNpy(REF + `win${wi}.npy`).data;
  for (const [tag, img] of [['same input', { data: refImg.data, dims: size }], ['js preprocessing', jsImg]]) {
    const { label } = await vision(R, extract(img, starts[wi], ROI), ROI, () => {}, () => {}, '');
    compare(`window ${wi + 1} at ${starts[wi]} (${tag})`, label, want);
  }
}
