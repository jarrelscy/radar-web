// Drive the page in Playwright's Chromium: node tools/browser-test.mjs [wasm|webgpu|auto] [input]
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_CORE || '/home/jarrelscy/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core');
const [device = 'wasm', input = '/tmp/radar_dcm/demo.zip', url = 'http://localhost:8765/'] = process.argv.slice(2);
const browser = await chromium.launch({
  headless: true,
  args: [...(process.env.SWIFTSHADER ? ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-watchdog'] : []), '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (!t.startsWith('[radar]')) console.log('console:', t.slice(0, 300)); });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(url);
await page.waitForFunction(() => window.crossOriginIsolated, null, { timeout: 15000 }).catch(() => console.log('not cross-origin isolated'));
console.log('caps:', await page.textContent('#caps').catch(() => ''));
await page.click(`label:has(input[name=device][value=${device}])`);
await page.waitForTimeout(500); console.log('caps:', await page.textContent('#caps'));
await page.setInputFiles('#files', input);
const t0 = Date.now();
await page.waitForFunction(() => !document.getElementById('results').hidden || /Error/.test(document.getElementById('stage').textContent), null, { timeout: 30 * 60e3 });
console.log((await page.textContent('#log')).trim());
console.log('summary:', await page.textContent('#summary'));
const top = await page.$$eval('#table .row', rs => rs.slice(0, 6).map(r => r.innerText.replace(/\s+/g, ' ')));
console.log(top.join('\n'));
const scores = await page.evaluate(() => window.__last);
console.log(`wall ${(Date.now() - t0) / 1000}s`);
await browser.close();
