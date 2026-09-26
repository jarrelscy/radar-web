// Uniform view over what the user dropped: a zip, loose files or a folder.
// Each entry is {name, size, head(n) -> Uint8Array of the first n bytes, blob() -> Blob}.
// Zip reading only touches the central directory plus the members asked for,
// so a 10 GB zip costs a few MB of reads to scan.

export const MB = b => (b / 1e6).toFixed(b < 1e8 ? 1 : 0) + ' MB';

export async function pmap(xs, fn, k = 16) {
  const out = new Array(xs.length); let i = 0;
  await Promise.all(Array.from({ length: Math.min(k, xs.length) }, async () => {
    while (i < xs.length) { const j = i++; out[j] = await fn(xs[j], j); }
  }));
  return out;
}

export const SKIP = /(^|\/)(__MACOSX\/|\._)|(^|\/)DICOMDIR$|\.(txt|xml|html?|jpe?g|png|pdf|exe|dll|inf|ini|json|md|csv)$/i;

async function readAt(f, off, len) { return new DataView(await f.slice(off, off + len).arrayBuffer()); }

async function streamHead(stream, n) {
  const r = stream.getReader(), parts = []; let got = 0;
  try { while (got < n) { const { done, value } = await r.read(); if (done) break; parts.push(value); got += value.length; } }
  catch (e) { if (!got) throw e; }
  finally { r.cancel().catch(() => {}); }
  const b = new Uint8Array(Math.min(got, n)); let o = 0;
  for (const p of parts) { const q = p.subarray(0, b.length - o); b.set(q, o); o += q.length; if (o >= b.length) break; }
  return b;
}

export async function zipEntries(f, log = () => {}) {
  const tl = Math.min(f.size, 65557), t = await readAt(f, f.size - tl, tl);
  let e = -1; for (let i = tl - 22; i >= 0; i--) if (t.getUint32(i, true) === 0x06054b50) { e = i; break; }
  if (e < 0) throw new Error('not a zip file');
  let count = t.getUint16(e + 10, true), cdSize = t.getUint32(e + 12, true), cdOff = t.getUint32(e + 16, true);
  let zip64 = false;
  if (count === 0xffff || cdSize === 0xffffffff || cdOff === 0xffffffff) {   // zip64: locator sits before the EOCD
    zip64 = true;
    if (e < 20 || t.getUint32(e - 20, true) !== 0x07064b50) throw new Error('bad zip64 record');
    const z = await readAt(f, Number(t.getBigUint64(e - 12, true)), 56);
    count = Number(z.getBigUint64(32, true)); cdSize = Number(z.getBigUint64(40, true)); cdOff = Number(z.getBigUint64(48, true));
  }
  log(`zip ${MB(f.size)}${zip64 ? ' (zip64)' : ''}: ${count} entries, central directory ${MB(cdSize)}`);
  const cd = await readAt(f, cdOff, cdSize), dec = new TextDecoder(), out = [];
  for (let k = 0, p = 0; k < count && p + 46 <= cdSize; k++) {
    if (cd.getUint32(p, true) !== 0x02014b50) break;
    const method = cd.getUint16(p + 10, true), nl = cd.getUint16(p + 28, true), xl = cd.getUint16(p + 30, true), cl = cd.getUint16(p + 32, true);
    let csize = cd.getUint32(p + 20, true), usize = cd.getUint32(p + 24, true), off = cd.getUint32(p + 42, true);
    const name = dec.decode(new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nl));
    for (let x = p + 46 + nl; x + 4 <= p + 46 + nl + xl; x += 4 + cd.getUint16(x + 2, true)) {   // zip64 extra: overflowed fields in order
      if (cd.getUint16(x, true) !== 1) continue; let q = x + 4;
      if (usize === 0xffffffff) { usize = Number(cd.getBigUint64(q, true)); q += 8; }
      if (csize === 0xffffffff) { csize = Number(cd.getBigUint64(q, true)); q += 8; }
      if (off === 0xffffffff) { off = Number(cd.getBigUint64(q, true)); }
    }
    p += 46 + nl + xl + cl;
    if (!name.endsWith('/') && !SKIP.test(name)) out.push(zipMember(f, { name, method, csize, usize, off }));
  }
  log(`kept ${out.length} candidate files (skipped ${count - out.length} folders/non-DICOM by name)`);
  return out;
}

function zipMember(f, m) {
  let data;
  const start = async () => {
    if (data === undefined) {
      const h = await readAt(f, m.off, 30);
      if (h.getUint32(0, true) !== 0x04034b50) throw new Error('bad zip entry ' + m.name);
      data = m.off + 30 + h.getUint16(26, true) + h.getUint16(28, true);
    }
    return data;
  };
  const stream = async () => {
    if (m.method !== 0 && m.method !== 8) throw new Error(`zip compression method ${m.method} not supported (${m.name})`);
    const d = await start(), s = f.slice(d, d + m.csize).stream();
    return m.method === 0 ? s : s.pipeThrough(new DecompressionStream('deflate-raw'));
  };
  return {
    name: m.name, size: m.usize,
    async head(n) {
      if (m.method === 0) { const d = await start(); return new Uint8Array(await f.slice(d, d + Math.min(n, m.csize)).arrayBuffer()); }
      return streamHead(await stream(), n);
    },
    async blob() {
      if (m.method === 0) { const d = await start(); return f.slice(d, d + m.csize); }
      return new Response(await stream()).blob();
    },
  };
}

export function fileEntry(file, name) {
  name = name || file.webkitRelativePath || file.name;
  return {
    name, size: file.size,
    async head(n) { return new Uint8Array(await file.slice(0, n).arrayBuffer()); },
    async blob() { return file; },
  };
}

// files: File[] (from <input>, drag and drop, or fs.openAsBlob in Node).
// Returns {kind:'dicom', entries} or {kind:'nifti', file}.
export async function openInput(files, log = () => {}) {
  files = [...files];
  const lower = f => (f.webkitRelativePath || f.name || '').toLowerCase();
  const nii = files.find(f => /\.nii(\.gz)?$/.test(lower(f)));
  if (nii) return { kind: 'nifti', file: nii, name: nii.name };
  const entries = [];
  for (const f of files) {
    if (lower(f).endsWith('.zip')) entries.push(...await zipEntries(f, log));
    else if (!SKIP.test(lower(f))) entries.push(fileEntry(f));
  }
  if (files.length > 1 || !lower(files[0]).endsWith('.zip')) log(`${entries.length} files`);
  return { kind: 'dicom', entries };
}

// Drag and drop of folders: walk FileSystemEntry trees into Files with a path.
export async function filesFromDataTransfer(dt) {
  const items = [...dt.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!items.length) return [...dt.files];
  const out = [];
  const walk = async (e, path) => {
    if (e.isFile) {
      const f = await new Promise((res, rej) => e.file(res, rej));
      Object.defineProperty(f, 'webkitRelativePath', { value: path + f.name });
      out.push(f);
    } else if (e.isDirectory) {
      const r = e.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => r.readEntries(res, rej));
        if (!batch.length) break;
        for (const c of batch) await walk(c, path + e.name + '/');
      }
    }
  };
  for (const e of items) await walk(e, '');
  return out;
}
