import fs from 'node:fs';
import { openInput } from '../lib/source.js';
import { pickSeries, seriesVolume } from '../lib/series.js';
import { preprocess } from '../lib/preprocess.js';
import { readNpy } from './npy.mjs';
const [zip = '/tmp/radar_dcm/demo.zip', ref = '/tmp/radar_ref/image.npy'] = process.argv.slice(2);
const log = m => console.log(m);
const file = new File([await fs.openAsBlob(zip)], zip.split('/').pop());
const inp = await openInput([file], log);
const pick = await pickSeries(inp.entries, log);
const vol = await seriesVolume(pick.best.files, log);
const img = await preprocess(vol, log);
const r = readNpy(ref);
console.log('js', img.dims, 'ref', r.shape);
if (img.dims.join() === r.shape.join()) {
  let md = 0, sum = 0;
  for (let i = 0; i < r.data.length; i++) { const d = Math.abs(r.data[i] - img.data[i]); sum += d; if (d > md) md = d; }
  console.log('max abs diff', md, 'mean', sum / r.data.length);
}
