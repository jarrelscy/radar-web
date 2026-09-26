// Full pipeline in Node with onnxruntime-web (wasm), compared with the
// Python reference (tools/reference.py) and the radar-svc scores.
//   node tools/node-test.mjs [input.zip|.nii.gz] [reference.json] [server.json]
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { openInput } from '../lib/source.js';
import { pickSeries, seriesVolume } from '../lib/series.js';
import { niftiVolume } from '../lib/nifti.js';
import { preprocess } from '../lib/preprocess.js';
import { loadRadar, runRadar } from '../lib/radar.js';
import { ENGLISH } from '../lib/labels.js';

const require = createRequire(new URL('./package.json', import.meta.url));
const ort = require('onnxruntime-web');
ort.env.wasm.numThreads = Number(process.env.THREADS || 16);
const [input = '/tmp/radar_dcm/demo.zip', refPath = '/tmp/radar_ref/reference.json', srvPath = '/tmp/radar_zip.json'] = process.argv.slice(2);
const t0 = performance.now();
const log = m => console.log(`[${((performance.now() - t0) / 1000).toFixed(1).padStart(6)}s] ${m}`);
const models = new URL('../models/', import.meta.url);
const R = await loadRadar(ort, { load: async n => new Uint8Array(fs.readFileSync(new URL(n, models))), device: 'wasm', log });
const file = new File([await fs.openAsBlob(input)], input.split('/').pop());
const inp = await openInput([file], log);
const vol = inp.kind === 'nifti' ? await niftiVolume(inp.file, log) : await seriesVolume((await pickSeries(inp.entries, log)).best.files, log);
const img = await preprocess(vol, log);
const res = await runRadar(R, img, { log });
fs.writeFileSync('/tmp/radar_web_out.json', JSON.stringify(res, null, 1));
const cmp = (name, ref) => {
  let md = 0, worst = '';
  for (const [k, v] of Object.entries(ref)) { const d = Math.abs((res.scores[k] ?? NaN) - v); if (!(d <= md)) { md = d; worst = k; } }
  console.log(`${name}: ${Object.keys(res.scores).length} vs ${Object.keys(ref).length} scores, max |diff| ${md.toFixed(5)} (${ENGLISH[worst] || worst})`);
};
if (fs.existsSync(refPath)) {
  const ref = JSON.parse(fs.readFileSync(refPath));
  console.log('ref windows ', JSON.stringify(ref.windows.map(w => [w.start, w.organs])));
  console.log('js  windows ', JSON.stringify(res.trace.windows.map(w => [w.start, w.organs])));
  console.log('ref fallback', JSON.stringify(ref.fallback.map(f => [f.item, f.box, f.organs])));
  console.log('js  fallback', JSON.stringify(res.trace.fallback.map(f => [f.item, f.box, f.organs])));
  cmp('vs reference', ref.scores);
}
if (fs.existsSync(srvPath)) cmp('vs radar-svc', Object.fromEntries(JSON.parse(fs.readFileSync(srvPath)).findings.map(f => [f.key, f.prob])));
for (const [k, v] of Object.entries(res.scores).sort((a, b) => b[1] - a[1]).slice(0, 8)) console.log(`  ${v.toFixed(4)}  ${ENGLISH[k]}`);
