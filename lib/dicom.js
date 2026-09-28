// Minimal DICOM reader: enough of the header to pick a series and place each
// slice, plus pixel data decoding. Native: uncompressed LE, deflated, RLE.
// JPEG lossless, JPEG-LS, JPEG 2000 and HTJ2K use vendored codecs loaded on demand.

const LONGVR = new Set(['OB', 'OW', 'OF', 'SQ', 'UT', 'UN', 'UC', 'UR', 'OD', 'OL', 'OV', 'SV', 'UV']);
const US_TAGS = new Set([0x00280002, 0x00280010, 0x00280011, 0x00280100, 0x00280101, 0x00280103]);
export const TS = {
  IMPLICIT: '1.2.840.10008.1.2', EXPLICIT: '1.2.840.10008.1.2.1', DEFLATED: '1.2.840.10008.1.2.1.99',
  BIG: '1.2.840.10008.1.2.2', RLE: '1.2.840.10008.1.2.5',
};

// Tags used to group files into series (same set as radar-svc _dicom_header).
export const SELECT_TAGS = {
  0x00080008: 'image_type', 0x00080060: 'modality', 0x0008103E: 'desc',
  0x00180015: 'body_part', 0x0020000E: 'series_uid', 0x00200037: 'iop',
};
// Tags needed to place and decode a slice. All sit before the pixel data.
export const SLICE_TAGS = {
  ...SELECT_TAGS,
  0x00180050: 'thickness', 0x00200013: 'instance', 0x00200032: 'ipp',
  0x00280002: 'samples', 0x00280008: 'frames', 0x00280010: 'rows', 0x00280011: 'cols',
  0x00280030: 'spacing', 0x00280100: 'bits', 0x00280101: 'bits_stored', 0x00280103: 'pixel_rep',
  0x00281052: 'intercept', 0x00281053: 'slope',
};
const PIXEL = 0x7FE00010;

// Parse a little-endian dataset from the start of a file (or a partial head).
// Stops after `stop` (a tag number) is passed. Returns [tags|null, complete];
// complete=false means the buffer ran out before reaching `stop`.
// With stop=PIXEL the pixel element is described as tags.pixel =
// {offset, length} for native data or {fragments:[[off,len],...]} if encapsulated.
// If `trace` is an array, every element read is pushed to it and trace.end says why parsing stopped.
export function parseDicom(b, want = SELECT_TAGS, stop = 0x00200037, trace = null) {
  const n = b.length, dv = new DataView(b.buffer, b.byteOffset, n);
  const str = (o, l) => { let s = ''; for (let i = 0; i < l && o + i < n; i++) s += String.fromCharCode(b[o + i]); return s; };
  let pos, explicit;
  if (n >= 132 && str(128, 4) === 'DICM') { pos = 132; explicit = true; }
  else if (n >= 6 && /^[A-Z]{2}$/.test(str(4, 2))) { pos = 0; explicit = true; }
  else if (n >= 8) { pos = 0; explicit = false; } else return [null, n === 0];
  // stack of the explicit flag outside each undefined-length element: a UN element of
  // undefined length holds an implicit VR sequence even in an explicit VR file
  let inMeta = pos === 132, depth = 0; const out = {}, outer = [];
  const partial = () => { if (trace) trace.end = { pos, why: 'ran out of data' }; return [Object.keys(out).length ? out : null, false]; };
  for (;;) {
    if (pos + 8 > n) return partial();
    const g = dv.getUint16(pos, true), e = dv.getUint16(pos + 2, true), tag = ((g << 16) | e) >>> 0;
    if (g === 0xFFFE) {
      const ln = dv.getUint32(pos + 4, true);
      trace?.push({ pos, tag, vr: '', ln, depth, explicit }); pos += 8;
      if (e === 0xE000 && ln !== 0xFFFFFFFF) pos += ln;
      else if (e === 0xE0DD && depth > 0) { depth--; explicit = outer.pop(); }
      continue;
    }
    if (inMeta && g !== 2) {
      inMeta = false; out.ts = out.ts || TS.EXPLICIT;
      if (out.ts === TS.BIG) return [null, true];
      if (out.ts === TS.DEFLATED) { out.deflated_at = pos; return [out, true]; }
      explicit = out.ts !== TS.IMPLICIT;
    }
    let ln, hdr, vr = '';
    if (explicit || inMeta) {
      vr = str(pos + 4, 2);
      if (LONGVR.has(vr)) { if (pos + 12 > n) return partial(); ln = dv.getUint32(pos + 8, true); hdr = 12; }
      else { ln = dv.getUint16(pos + 6, true); hdr = 8; }
    } else { ln = dv.getUint32(pos + 4, true); hdr = 8; }
    trace?.push({ pos, tag, vr, ln, depth, explicit: explicit || inMeta });
    if (depth === 0 && tag > stop) { if (trace) trace.end = { pos, why: `tag ${hex8(tag)} is past the stop tag` }; return [out, true]; }
    if (depth === 0 && tag === PIXEL) {
      if (ln !== 0xFFFFFFFF) { out.pixel = { offset: pos + hdr, length: ln }; return [out, pos + hdr + ln <= n]; }
      const frags = []; let p = pos + hdr, first = true;
      for (;;) {
        if (p + 8 > n) return partial();
        const ig = dv.getUint16(p, true), ie = dv.getUint16(p + 2, true), il = dv.getUint32(p + 4, true);
        if (ig !== 0xFFFE || ie === 0xE0DD) break;
        if (!first) frags.push([p + 8, il]);
        first = false; p += 8 + il;
      }
      out.pixel = { fragments: frags }; return [out, p <= n];
    }
    if (ln === 0xFFFFFFFF) { depth++; outer.push(explicit); if (vr === 'UN') explicit = false; pos += hdr; continue; }
    if (depth === 0 && (want[tag] || tag === 0x00020010)) {
      if (pos + hdr + ln > n) return partial();
      if (US_TAGS.has(tag) && (vr === 'US' || vr === 'SS' || !vr) && ln >= 2) out[want[tag]] = dv.getUint16(pos + hdr, true);
      else out[tag === 0x00020010 ? 'ts' : want[tag]] = str(pos + hdr, ln).replace(/[\0 ]+$/, '').trim();
      if (tag === stop) return [out, true];
    }
    pos += hdr + ln;
  }
}

const hex8 = t => '(' + (t >>> 16).toString(16).padStart(4, '0') + ',' + (t & 0xFFFF).toString(16).padStart(4, '0') + ')';
const ITEM = { 0xE000: 'item', 0xE00D: 'item end', 0xE0DD: 'sequence end' };
const SAFE_VR = new Set(['US', 'SS', 'UL', 'SL', 'CS', 'DS', 'IS', 'UI', 'FL', 'FD']);

// Readable listing of a file's element structure for debugging. Values are shown only for
// numeric, code and UID elements outside the patient group, so names and dates are not printed.
export function dumpDicom(b, name = '') {
  const trace = [], [t] = parseDicom(b, SLICE_TAGS, PIXEL, trace);
  const dv = new DataView(b.buffer, b.byteOffset, b.length), lines = [`DICOM structure of ${name} (${b.length} bytes, transfer syntax ${t?.ts || '?'}):`];
  for (const r of trace.slice(0, 600)) {
    const pad = '  '.repeat(r.depth), g = r.tag >>> 16;
    let what = ITEM[r.tag & 0xFFFF] && g === 0xFFFE ? ITEM[r.tag & 0xFFFF] : r.vr || (r.explicit ? '??' : 'implicit');
    let val = '';
    if (r.vr && SAFE_VR.has(r.vr) && g !== 0x0010 && r.ln <= 64 && r.ln !== 0xFFFFFFFF) {
      const o = r.pos + 8;
      if (r.vr === 'US' && r.ln >= 2) val = String(dv.getUint16(o, true));
      else if (r.vr === 'SS' && r.ln >= 2) val = String(dv.getInt16(o, true));
      else if ((r.vr === 'UL' || r.vr === 'SL') && r.ln >= 4) val = String(dv.getUint32(o, true));
      else if (r.vr === 'FL' || r.vr === 'FD') val = '';
      else val = JSON.stringify(Array.from(b.subarray(o, o + r.ln), c => String.fromCharCode(c)).join('').replace(/\0/g, ''));
    }
    lines.push(`${String(r.pos).padStart(8)} ${pad}${hex8(r.tag)} ${what.padEnd(8)} len ${r.ln === 0xFFFFFFFF ? 'undefined' : r.ln}${val ? ' = ' + val : ''}`);
  }
  if (trace.length > 600) lines.push(`... ${trace.length - 600} more elements`);
  const end = trace.end || { pos: -1, why: t?.pixel ? 'reached pixel data' : 'finished' };
  lines.push(`stopped: ${end.why}${end.pos >= 0 ? ' at byte ' + end.pos : ''}`);
  if (end.pos >= b.length - 8 && trace.length) end.pos = trace[trace.length - 1].pos;   // bytes of the element whose length ran past the end
  if (end.pos >= 0) {
    const a = Math.max(0, end.pos - 32), bytes = Array.from(b.subarray(a, end.pos + 32), x => x.toString(16).padStart(2, '0'));
    lines.push(`bytes ${a}-${a + bytes.length - 1}: ${bytes.join(' ')}`);
  }
  for (let p = b.length - 4; p >= 0; p--) if (b[p] === 0xE0 && b[p + 1] === 0x7F && b[p + 2] === 0x10 && b[p + 3] === 0) {
    lines.push(`last (7fe0,0010) pattern at byte ${p}: ${Array.from(b.subarray(p, p + 16), x => x.toString(16).padStart(2, '0')).join(' ')}`); break;
  }
  return lines.join('\n');
}

async function inflateRaw(bytes) {
  const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

// Header of one file for series grouping, reading as little as possible.
export async function selectHeader(entry) {
  for (let n = 16384; ; n *= 4) {
    const b = await entry.head(n), [t, c] = parseDicom(b, SELECT_TAGS);
    if (c || b.length < n || n >= (8 << 20)) return t && t.series_uid ? t : null;
  }
}

// Placement header (everything up to the pixel data) without reading pixels.
// no_pixels is set when the whole file was read without reaching pixel data (header-only files).
export async function sliceHeader(entry) {
  for (let n = 16384; ; n *= 4) {
    let b = await entry.head(n), [t, c] = parseDicom(b, SLICE_TAGS, 0x7FDFFFFF);
    if (t && t.deflated_at !== undefined) {   // whole dataset is deflated; need the full file
      b = new Uint8Array(await (await entry.blob()).arrayBuffer());
      const body = await inflateRaw(b.subarray(t.deflated_at));
      [t, c] = parseDicom(body, SLICE_TAGS, 0x7FDFFFFF); if (t && !c) t.no_pixels = true; return t;
    }
    if (c || b.length < n || n >= (8 << 20)) { if (t && !c && b.length < n) t.no_pixels = true; return t; }
  }
}

export const nums = s => (s || '').split('\\').map(Number);

// ---- pixel decoding
const codecs = {};
let codecBase = new URL('../vendor/codecs/', import.meta.url).href;
export function setCodecBase(url) { codecBase = url; }

async function fetchBytes(url) {
  if (typeof process !== 'undefined' && process.versions?.node && url.startsWith('file:')) {
    const fs = await import('node:fs/promises'); return new Uint8Array(await fs.readFile(new URL(url)));
  }
  return new Uint8Array(await (await fetch(url)).arrayBuffer());
}
// The cornerstone codecs are emscripten UMD scripts; run them with a fake `module`.
async function loadEmscripten(js, wasm) {
  const src = new TextDecoder().decode(await fetchBytes(codecBase + js));
  const module = { exports: {} };
  new Function('module', 'exports', 'require', '__filename', src)(module, module.exports, undefined, undefined);
  const factory = module.exports.default || module.exports;
  return factory({ wasmBinary: await fetchBytes(codecBase + wasm) });
}
function codec(name) {
  if (!codecs[name]) codecs[name] = (async () => {
    if (name === 'lossless') return (await import(codecBase + 'lossless.js'));
    if (name === 'jls') return loadEmscripten('charlswasm_decode.js', 'charlswasm_decode.wasm');
    if (name === 'j2k') return loadEmscripten('openjpegwasm_decode.js', 'openjpegwasm_decode.wasm');
    if (name === 'htj2k') return loadEmscripten('openjphjs.js', 'openjphjs.wasm');
  })();
  return codecs[name];
}
function codecFor(ts) {
  if (ts === '1.2.840.10008.1.2.4.57' || ts === '1.2.840.10008.1.2.4.70') return 'lossless';
  if (ts === '1.2.840.10008.1.2.4.80' || ts === '1.2.840.10008.1.2.4.81') return 'jls';
  if (ts === '1.2.840.10008.1.2.4.90' || ts === '1.2.840.10008.1.2.4.91') return 'j2k';
  if (/^1\.2\.840\.10008\.1\.2\.4\.20[123]$/.test(ts)) return 'htj2k';
  return null;
}
export function tsSupported(ts) {
  return !ts || ts === TS.IMPLICIT || ts === TS.EXPLICIT || ts === TS.DEFLATED || ts === TS.RLE || !!codecFor(ts);
}

function decodeRLE(frame, rows, cols, bytesPer) {
  const dv = new DataView(frame.buffer, frame.byteOffset, frame.length);
  const nseg = dv.getUint32(0, true), npx = rows * cols, out = new Uint8Array(npx * bytesPer);
  for (let s = 0; s < nseg && s < bytesPer; s++) {
    const start = dv.getUint32(4 + 4 * s, true), end = s + 1 < nseg ? dv.getUint32(8 + 4 * s, true) : frame.length;
    // segment s holds byte (bytesPer-1-s) of each little-endian sample
    let p = start, o = bytesPer - 1 - s, i = 0;
    while (p < end && i < npx) {
      const c = (frame[p++] << 24) >> 24;
      if (c >= 0) { for (let k = 0; k <= c && i < npx; k++, i++) out[i * bytesPer + o] = frame[p++]; }
      else if (c > -128) { const v = frame[p++]; for (let k = 0; k <= -c && i < npx; k++, i++) out[i * bytesPer + o] = v; }
    }
  }
  return out;
}

async function decodeWasm(mod, cls, bytes) {
  const dec = new mod[cls]();
  try {
    dec.getEncodedBuffer(bytes.length).set(bytes);
    dec.decode();
    const info = dec.getFrameInfo();
    return { bytes: dec.getDecodedBuffer().slice(), bits: info.bitsPerSample, signed: info.isSigned };
  } finally { dec.delete(); }
}

let dumped = false;

// Fallback when the dataset walk loses its place (for example in malformed private tags):
// look for the pixel data element from the end of the file.
function findPixel(b, implicit) {
  const dv = new DataView(b.buffer, b.byteOffset, b.length);
  for (let p = b.length - 12; p >= 0; p--) {
    if (b[p] !== 0xE0 || b[p + 1] !== 0x7F || b[p + 2] !== 0x10 || b[p + 3] !== 0x00) continue;
    const vr = String.fromCharCode(b[p + 4], b[p + 5]);
    const ex = !implicit && (vr === 'OB' || vr === 'OW' || vr === 'UN') && b[p + 6] === 0 && b[p + 7] === 0;
    const hdr = ex ? 12 : 8, ln = dv.getUint32(p + (ex ? 8 : 4), true);
    if (ln === 0xFFFFFFFF) {
      const frags = []; let q = p + hdr, first = true;
      while (q + 8 <= b.length && dv.getUint16(q, true) === 0xFFFE && dv.getUint16(q + 2, true) === 0xE000) {
        const il = dv.getUint32(q + 4, true);
        if (!first) frags.push([q + 8, il]);
        first = false; q += 8 + il;
      }
      const end = q + 8 <= b.length && dv.getUint16(q, true) === 0xFFFE && dv.getUint16(q + 2, true) === 0xE0DD;
      if (frags.length && end) return { fragments: frags };
    } else if (ln >= 1024 && p + hdr + ln <= b.length && b.length - (p + hdr + ln) <= 1024) return { offset: p + hdr, length: ln };
  }
  return null;
}

// Decode one single-frame, single-sample slice to Float32 HU (slope/intercept applied).
export async function decodeSlice(entry, hdr) {
  let b = new Uint8Array(await (await entry.blob()).arrayBuffer());
  let [t] = parseDicom(b, SLICE_TAGS, PIXEL);
  if (t && t.deflated_at !== undefined) { b = await inflateRaw(b.subarray(t.deflated_at)); [t] = parseDicom(b, SLICE_TAGS, PIXEL); }
  if (t && !t.pixel) t.pixel = findPixel(b, t.ts === TS.IMPLICIT);
  if (!t || !t.pixel) {
    if (!dumped) { dumped = true; console.log(dumpDicom(b, entry.name)); }
    throw new Error(`no pixel data found in ${entry.name} (the file structure is printed in the browser console)`);
  }
  hdr = { ...t, ...hdr };
  const rows = hdr.rows, cols = hdr.cols, npx = rows * cols;
  const bits = hdr.bits || 16, signed = hdr.pixel_rep === 1;
  if ((hdr.samples || 1) !== 1) throw new Error('colour images are not supported');
  const ts = t.ts || TS.IMPLICIT;
  let raw;   // little-endian sample bytes
  const frame = () => {
    const fr = t.pixel.fragments; if (fr.length === 1) return b.subarray(fr[0][0], fr[0][0] + fr[0][1]);
    const len = fr.reduce((a, f) => a + f[1], 0), o = new Uint8Array(len); let p = 0;
    for (const [off, l] of fr) { o.set(b.subarray(off, off + l), p); p += l; }
    return o;
  };
  if (t.pixel.fragments) {
    if (ts === TS.RLE) raw = decodeRLE(frame(), rows, cols, bits / 8);
    else {
      const name = codecFor(ts);
      if (!name) throw new Error(`transfer syntax ${ts} not supported`);
      const mod = await codec(name), f = frame();
      if (name === 'lossless') {
        const d = new mod.Decoder().decode(f.buffer, f.byteOffset, f.length, bits / 8);
        raw = new Uint8Array(d.buffer, d.byteOffset, d.byteLength);
      } else {
        const r = await decodeWasm(mod, name === 'jls' ? 'JpegLSDecoder' : name === 'j2k' ? 'J2KDecoder' : 'HTJ2KDecoder', f);
        raw = r.bytes;
      }
    }
  } else raw = b.subarray(t.pixel.offset, t.pixel.offset + t.pixel.length);

  const out = new Float32Array(npx);
  const slope = hdr.slope !== undefined && hdr.slope !== '' ? Number(hdr.slope) : 1;
  const icpt = hdr.intercept !== undefined && hdr.intercept !== '' ? Number(hdr.intercept) : 0;
  const bytesPer = raw.length >= npx * 2 && bits >= 16 ? (bits === 32 ? 4 : 2) : 1;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const stored = hdr.bits_stored || bits, shift = 32 - stored;
  for (let i = 0; i < npx; i++) {
    let v;
    if (bytesPer === 2) v = dv.getUint16(2 * i, true);
    else if (bytesPer === 4) v = dv.getUint32(4 * i, true);
    else v = raw[i];
    if (stored < 32) v = signed ? (v << shift) >> shift : (v << shift) >>> shift;
    out[i] = v * slope + icpt;
  }
  return out;
}
