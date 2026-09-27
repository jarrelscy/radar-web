import { filesFromDataTransfer } from './lib/source.js';
import { ENGLISH, TEST_ITEMS } from './lib/labels.js';

const $ = id => document.getElementById(id);
const logEl = $('log');
const t0 = performance.now();
let worker = null, last = null, busy = false, runStart = 0, cur = { name: '', frac: 0 };

function log(msg) {
  const t = ((performance.now() - t0) / 1000).toFixed(2).padStart(7);
  logEl.textContent += `[${t}s] ${msg}\n`; logEl.scrollTop = logEl.scrollHeight;
  console.log('[radar]', msg);
}

// capability line
(async () => {
  const bits = [];
  const gpu = navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null);
  bits.push(gpu ? 'WebGPU available' : 'no WebGPU');
  bits.push(crossOriginIsolated ? `CPU: ${Math.min(navigator.hardwareConcurrency || 4, 16)} threads` : 'CPU: 1 thread (page not cross-origin isolated)');
  $('caps').textContent = bits.join(' · ');
})();

function device() { return document.querySelector('input[name=device]:checked').value; }

function startWorker() {
  if (worker) return;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onMessage;
  worker.onerror = e => { log(`worker error: ${e.message}`); done(); };
}

// Fetch and build the models as soon as the page opens so the first scan starts straight away.
startWorker();
$('status').hidden = false; setStage('Loading models', 0);
log('loading models');
worker.postMessage({ preload: true, device: device() });
document.querySelectorAll('input[name=device]').forEach(el => el.addEventListener('change', () => { if (!busy) worker.postMessage({ preload: true, device: device() }); }));

function run(files) {
  files = [...files];
  if (!files.length || busy) return;
  busy = true; runStart = performance.now();
  $('drop').classList.add('busy'); $('status').hidden = false; $('results').hidden = true;
  log('---- new scan');
  setStage('Starting', 0);
  log(`${files.length} file(s): ${files.slice(0, 3).map(f => f.webkitRelativePath || f.name).join(', ')}${files.length > 3 ? ', …' : ''}`);
  startWorker();
  worker.postMessage({ files, device: device() });
}

function done() { busy = false; $('drop').classList.remove('busy'); }

function setStage(name, frac) {
  cur = { name, frac };
  $('stage').textContent = name;
  $('bar').style.width = `${Math.round(frac * 100)}%`;
  showPct();
}

// Elapsed seconds keep ticking during long single steps (the middle network is one call of several seconds).
function showPct() {
  const pct = cur.frac > 0 ? `${Math.round(cur.frac * 100)}%` : '';
  $('pct').textContent = busy ? [pct, `${((performance.now() - runStart) / 1000).toFixed(0)}s`].filter(Boolean).join(' · ') : pct;
}
setInterval(() => { if (busy) showPct(); }, 1000);

function onMessage({ data }) {
  if (data.type === 'log') log(data.msg);
  else if (data.type === 'ready') { if (!busy) setStage('Models ready. Drop a scan to start.', 1); }
  else if (data.type === 'progress') setStage(data.stage, data.frac);
  else if (data.type === 'error') { log(`ERROR: ${data.msg}`); setStage(`Error: ${data.msg}`, 0); done(); }
  else if (data.type === 'result') { last = data; setStage(`Done in ${data.seconds.toFixed(0)}s`, 1); show(data); done(); }
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function show(r) {
  $('results').hidden = false;
  const i = r.info || {}, pos = Object.values(r.scores).filter(p => p >= 0.5).length;
  const sum = [];
  if (i.series_description !== undefined) sum.push(`Series <b>${esc(i.series_description || '?')}</b>${i.body_part ? ` [${esc(i.body_part)}]` : ''} (${esc(i.series_selection)})`);
  if (i.size) sum.push(`${i.size.join('×')} voxels at ${i.spacing_mm.join(' × ')} mm`);
  sum.push(`<b>${pos}</b> of ${Object.keys(r.scores).length} findings ≥ 50%`);
  sum.push(`${r.device === 'webgpu' ? 'WebGPU' : 'CPU'}, ${r.seconds.toFixed(0)}s`);
  sum.push('research use only, not for clinical decisions');
  $('summary').innerHTML = sum.join(' · ');
  render();
}

function render() {
  if (!last) return;
  const view = document.querySelector('input[name=view]:checked').value, posOnly = $('posOnly').checked;
  const rows = TEST_ITEMS.filter(k => k in last.scores).map(k => {
    const [organ, finding] = ENGLISH[k].split('_');
    return { k, organ, finding, p: last.scores[k] };
  }).filter(r => !posOnly || r.p >= 0.5);
  const row = (r, withOrgan) => `<div class="row${r.p >= 0.5 ? ' pos' : ''}" title="${esc(r.k)}">
    <span class="name">${withOrgan ? `<small>${esc(r.organ)}</small>` : ''}${esc(r.finding)}</span>
    <span class="rbar"><i style="width:${(r.p * 100).toFixed(1)}%"></i></span>
    <span class="pct">${(r.p * 100).toFixed(0)}%</span></div>`;
  let html = '';
  if (view === 'score') html = rows.sort((a, b) => b.p - a.p).map(r => row(r, true)).join('');
  else {
    const by = new Map();
    for (const r of rows) { if (!by.has(r.organ)) by.set(r.organ, []); by.get(r.organ).push(r); }
    const organs = [...by].sort((a, b) => Math.max(...b[1].map(r => r.p)) - Math.max(...a[1].map(r => r.p)));
    for (const [organ, rs] of organs) {
      const n = rs.filter(r => r.p >= 0.5).length;
      html += `<div class="organ">${esc(organ)} <small>${n ? `${n} positive` : ''}</small></div>` + rs.sort((a, b) => b.p - a.p).map(r => row(r, false)).join('');
    }
  }
  if (!rows.length) html = '<p class="missing">No findings at or above 50%.</p>';
  if (last.organsNotFound?.length) {
    const names = [...new Set(TEST_ITEMS.filter(k => last.organsNotFound.includes(k.split('_')[0])).map(k => ENGLISH[k].split('_')[0]))];
    html += `<p class="missing">Not found in this volume (no score): ${esc(names.join(', '))}</p>`;
  }
  $('table').innerHTML = html;
}

document.querySelectorAll('input[name=view], #posOnly').forEach(el => el.addEventListener('change', render));
$('download').onclick = () => {
  if (!last) return;
  const findings = Object.entries(last.scores).sort((a, b) => b[1] - a[1]).map(([key, prob]) => {
    const [organ, finding] = ENGLISH[key].split('_'); return { key, organ, finding, prob: +prob.toFixed(4) };
  });
  const blob = new Blob([JSON.stringify({ disclaimer: 'Research use only. Not a medical device and not for diagnosis or treatment decisions.', model: 'RADAR (Alibaba DAMO Academy), CC BY-NC-SA 4.0', ...last.info, device: last.device, seconds: +last.seconds.toFixed(1), findings, organs_not_found: last.organsNotFound, trace: last.trace }, null, 1)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'radar-results.json' });
  a.click(); URL.revokeObjectURL(a.href);
};

$('files').onchange = e => { run(e.target.files); e.target.value = ''; };
$('folder').onchange = e => { run(e.target.files); e.target.value = ''; };
const drop = $('drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); run(await filesFromDataTransfer(e.dataTransfer)); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());
