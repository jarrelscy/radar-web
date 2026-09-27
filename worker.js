// Runs the whole pipeline off the main thread. Messages in: {files, device} or {preload: true, device}.
// Messages out: {type:'log'|'progress'|'result'|'error', ...}.
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { openInput } from './lib/source.js';
import { pickSeries, seriesVolume } from './lib/series.js';
import { niftiVolume } from './lib/nifti.js';
import { preprocess } from './lib/preprocess.js';
import { loadRadar, runRadar } from './lib/radar.js';

const base = new URL('./', import.meta.url).href;
ort.env.wasm.wasmPaths = base + 'vendor/ort/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
ort.env.logLevel = 'error';

// Model files are on Hugging Face, pinned to one commit so the browser cache never mixes versions.
const MODELS = 'https://huggingface.co/jarrelscy/radar-onnx/resolve/5108c03d75188062ac1b6841c3abab07afcff662/';
const CACHE = 'radar-models';
const post = (type, x = {}) => self.postMessage({ type, ...x });
const log = msg => post('log', { msg });
const stage = (name, frac) => post('progress', { stage: name, frac });

// Models come from the Cache API after the first visit.
async function loader() {
  if (self.caches) for (const k of await caches.keys()) if (k.startsWith('radar-models-')) await caches.delete(k);
  const cache = self.caches ? await caches.open(CACHE) : null;
  const sizes = { 'part_b.data0': 57698304, 'part_b.data1': 44425472 };
  const total = 116e6, MB = x => (x / 1e6).toFixed(1) + ' MB';
  let got = 0, fetched = false, logged = 0;
  return async name => {
    const url = MODELS + name, t0 = performance.now();
    let res = cache && await cache.match(url);
    if (!res) {
      if (!fetched) { fetched = true; log('downloading models from Hugging Face (~116 MB, cached in this browser for next time)'); }
      const r = await fetch(url).catch(() => { throw new Error(`could not reach Hugging Face for ${name}. The models need an internet connection the first time; after that they load from this browser's cache`); });
      if (!r.ok) throw new Error(`failed to fetch ${name} from Hugging Face: ${r.status}`);
      const parts = [], rd = r.body.getReader();
      let n = 0;
      for (;;) {
        const { done, value } = await rd.read(); if (done) break;
        parts.push(value); n += value.length; got += value.length;
        stage(`Downloading models · ${MB(got)} of ${MB(total)}`, Math.min(got / total, 1));
        if (got - logged >= 20e6) { logged = got; log(`downloaded ${MB(got)} of ${MB(total)}`); }
      }
      const blob = new Blob(parts);
      if (cache) await cache.put(url, new Response(blob)).catch(() => {});
      log(`${name}: ${MB(n)} downloaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      return new Uint8Array(await blob.arrayBuffer());
    }
    got += sizes[name] || 0;
    const buf = new Uint8Array(await res.arrayBuffer());
    log(`${name}: ${MB(buf.length)} from browser cache`);
    return buf;
  };
}

log(`cross-origin isolated: ${self.crossOriginIsolated}; CPU threads: ${ort.env.wasm.numThreads}`);

// One model load at a time; a preload started on page open is reused by the first run.
let radar = null;
function getRadar(device) {
  if (radar && radar.device === device) return radar.p;
  const old = radar;
  const p = (async () => {
    if (old) await old.p.then(r => Promise.all([r.sessA, r.sessB, r.sessC].map(x => x.release())), () => {});
    return loadRadar(ort, { load: await loader(), device, log });
  })();
  radar = { device, p };
  p.catch(() => { if (radar && radar.p === p) radar = null; });
  return p;
}

let chosen = {};
async function pickDeviceOnce(want) {
  return chosen[want] ??= pickDevice(want).catch(e => { delete chosen[want]; throw e; });
}

async function pickDevice(want) {
  if (want === 'wasm') return 'wasm';
  const gpu = self.navigator.gpu && await self.navigator.gpu.requestAdapter().catch(() => null);
  if (gpu) {
    const l = gpu.limits;
    log(`WebGPU adapter: maxBufferSize ${(l.maxBufferSize / 1e6).toFixed(0)} MB, maxStorageBufferBindingSize ${(l.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB`);
    return 'webgpu';
  }
  if (want === 'webgpu') throw new Error('WebGPU is not available in this browser');
  log('WebGPU not available; using CPU');
  return 'wasm';
}

let latest = 0;
self.onmessage = async ({ data }) => {
  const t0 = performance.now(), me = ++latest;
  if (data.preload) {
    try {
      stage('Loading models', 0);
      const device = await pickDeviceOnce(data.device);
      if (me !== latest) return; // a newer preload or a scan has taken over
      await getRadar(device);
      post('ready');
    } catch (e) { log(`model preload failed (${e.message}); will retry when a scan is loaded`); post('ready'); }
    return;
  }
  try {
    let device = await pickDeviceOnce(data.device);
    const modelsReady = getRadar(device).catch(e => e);

    stage('Reading input', 0);
    const inp = await openInput(data.files, log);
    let vol, info = {};
    if (inp.kind === 'nifti') vol = await niftiVolume(inp.file, log);
    else {
      const pick = await pickSeries(inp.entries, log);
      info = { series_description: pick.best.desc || '', body_part: pick.best.body_part || '', series_selection: pick.why, series_in_upload: pick.list };
      vol = await seriesVolume(pick.best.files, log);
    }
    info.size = vol.dims; info.spacing_mm = vol.spacing.map(s => +s.toFixed(3));
    stage('Preprocessing', 0);
    const img = await preprocess(vol, log, f => stage('Decoding slices', f));

    stage('Loading models', 0);
    let R = await modelsReady;
    if (R instanceof Error) {
      if (device !== 'webgpu' || data.device === 'webgpu') throw R;
      log(`WebGPU session failed (${R.message}); falling back to CPU`);
      device = 'wasm'; R = await getRadar(device);
    }
    let res;
    try {
      res = await runRadar(R, img, { log, progress: (f, d) => stage(`Running RADAR (${R.device === 'webgpu' ? 'WebGPU' : 'CPU'})${d ? ` · ${d}` : ''}`, f) });
    } catch (e) {
      if (R.device !== 'webgpu' || data.device === 'webgpu') throw e;
      log(`WebGPU run failed (${e.message}); retrying on CPU`);
      R = await getRadar('wasm');
      res = await runRadar(R, img, { log, progress: (f, d) => stage(`Running RADAR (CPU)${d ? ` · ${d}` : ''}`, f) });
    }
    log(`total ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    post('result', { ...res, info, device: R.device, seconds: (performance.now() - t0) / 1000 });
  } catch (e) {
    console.error(e);
    post('error', { msg: e && e.message ? e.message : String(e) });
  }
};
