# tools

Build and test scripts. None of this is needed to run the site.

- `export.py`: converts `checkpoint_radar_pretrain.pth` into `models/` (three ONNX parts plus `head.bin`/`head.json`)
  and checks the rewritten network against the original VisionBranch.
- `reference.py`: runs the radar-svc pipeline in Python on one input and dumps the preprocessed volume,
  the windows, which organs each window scored, the fallback crops and the scores.
- `node-test.mjs`: runs the browser pipeline in Node with onnxruntime-web (wasm) and compares it with `reference.py` output.
- `test-preprocess.mjs`: compares only the preprocessed volume.
- `browser-test.mjs`: drives the page in Playwright Chromium (`SWIFTSHADER=1` for the software WebGPU adapter).

```
uv venv .venv-conv -p 3.11 && uv pip install --python .venv-conv/bin/python torch onnx onnxruntime monai simpleitk nibabel onnxscript fastapi
.venv-conv/bin/python tools/export.py
.venv-conv/bin/python tools/reference.py case.zip /tmp/radar_ref
(cd tools && npm install) && node tools/node-test.mjs case.zip /tmp/radar_ref/reference.json
```

## WebGPU checks

- `parts.html` + `parts-test.mjs`: runs each ONNX part on WebGPU and CPU with the same random input and compares the outputs. `node tools/parts-test.mjs "d=96&h=256&w=384&only=b"` uses SwiftShader; set `GPU=1` to use the real GPU. Needs the static server on :8765.
- `cut.py`: cuts part_b at a named tensor into `cut.onnx`, which `parts.html?only=cut` runs on WebGPU. Used to find the op that fails.
