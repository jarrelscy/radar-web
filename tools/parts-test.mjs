// node tools/parts-test.mjs "d=32&h=128&w=128"   (SwiftShader WebGPU unless GPU=1)
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PW_CORE || '/home/jarrelscy/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core');
const query = process.argv[2] || '';
const browser = await chromium.launch({
  headless: true,
  args: [...(process.env.GPU ? [] : ['--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-watchdog']), '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu'],
});
const page = await browser.newPage();
page.on('console', m => { const t = m.text(); if (t.startsWith('[parts]')) console.log(t.slice(8)); else if (!/VerifyEachNode|Rerunning|Service Worker/.test(t)) console.log('console:', t.slice(0, 400)); });
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(`http://localhost:8765/tools/${process.env.PAGE || 'parts'}.html?${query}`);
await page.waitForFunction(() => /DONE/.test(document.getElementById('out')?.textContent || ''), null, { timeout: 60 * 60e3 });
await browser.close();
