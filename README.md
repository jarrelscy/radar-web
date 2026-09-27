# RADAR in the browser

**https://jarrelscy.github.io/radar-web/**

Runs [RADAR](https://github.com/alibaba-damo-academy/damo-radar), the generalist abdominal CT model from Alibaba DAMO Academy, entirely in the browser. Drop in a CT and it scores 146 findings across 18 organs. It uses WebGPU when available and otherwise the CPU (WebAssembly, multi-threaded).

Images never leave the computer. They are read and processed in the browser tab and nothing is uploaded. The model files (~116 MB) are downloaded from [Hugging Face](https://huggingface.co/jarrelscy/radar-onnx) on first use and cached in the browser.

> **Research use only.** This is not a medical device. It has not been validated or approved for clinical use and must not be used for diagnosis or treatment decisions. RADAR ships no calibrated thresholds; the page shows scores of 50% or more as positive.

## Inputs

- a zip of a DICOM study (zip64 is fine; only the zip index and the chosen series are read, so large zips are quick)
- a DICOM folder or loose DICOM files
- a `.nii` / `.nii.gz` volume

From a study, the largest axial CT series with "abd" in its Series Description or Body Part Examined is used, and otherwise the largest CT series. Scouts, localizers and series with fewer than 16 images are skipped. Supported DICOM transfer syntaxes are uncompressed, deflated, RLE, JPEG lossless, JPEG-LS, JPEG 2000 and HTJ2K. Enhanced multi-frame DICOM isn't supported yet.

## How it works

The pipeline follows `evaluate()` in the upstream `inference_demo.py` and the radar-svc CPU service:

1. The volume is reoriented to LAS and resized to 1 × 1 × 5 mm (trilinear). Only the source slices that feed the resized volume are decoded.
2. It is clipped to [-300, 400] HU, min-max normalised, cropped to the body and padded to at least 96 × 256 × 384.
3. Sliding windows of 96 × 256 × 384 with 25% overlap. Each organ is scored once, in the first window where its predicted mask doesn't touch the window edge. Any organ still unscored gets a window centred on it.
4. Scoring: the organ's query token attends over the image tokens covering the organ mask, and the result is compared with precomputed positive/negative text embeddings for each finding.

The vision network (a 3D nnU-Net-style encoder/decoder) was rewritten using only 2D convolutions, with depth treated as the batch axis. A 3×3×3 convolution becomes three shifted 2D convolutions, and BatchNorm is folded in. It is exported as three ONNX parts, so the full-resolution activations are only ever processed 8 slices at a time. The files are hosted at [jarrelscy/radar-onnx](https://huggingface.co/jarrelscy/radar-onnx) and loaded from a pinned commit:

| file | runs on | does |
|---|---|---|
| `part_a.onnx` | 8-slice slabs | stem and the first stage-2 conv |
| `part_b.onnx` + `.data0/.data1` | whole window at 1/4 in-plane size | rest of the encoder, decoder up to 1/4 size, token embeddings |
| `part_c.onnx` | 8-slice slabs | last decoder stage, segmentation head, upsample, argmax |
| `head.bin` | plain JS | attention head, projections, text embeddings |

On the upstream demo case (as DICOM zip and as NIfTI) the browser output matches the PyTorch pipeline: the same windows, the same organs scored in each, and all 146 scores within 1e-5. One known difference: upstream averages segmentation probabilities where windows overlap before placing the centred crops, while this port keeps the label from the last window. That can move a crop by a few voxels on some cases.

Performance on the demo case (2 windows plus 1 centred crop): about 40 s on a 16-thread CPU. If a WebGPU run fails in Auto mode, it retries on the CPU.

## Running locally

Any static file server works. The page registers `coi-serviceworker.js` to enable cross-origin isolation, which WebAssembly threads need:

```
python3 -m http.server 8000   # then open http://localhost:8000/
```

Models are fetched from Hugging Face and cached in the browser (Cache API) after the first load (~116 MB). `tools/` has the conversion and test scripts (see `tools/README.md`).

## Licence and attribution

The RADAR model weights and the derived ONNX files on Hugging Face are by Alibaba DAMO Academy and are redistributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This repository is released under the same licence (see `LICENSE`). Upstream code derives from LAVIS, nnU-Net, MONAI and 3D-ResNets-PyTorch. Bundled third-party libraries are listed in `THIRD_PARTY_LICENSES.md`.

```bibtex
@article{damo-radar-2026,
    author = {Qi Zhang and Jianpeng Zhang and Weiwei Cao and Zilin Lu and Wanxing Chang and Haonan Ding and Cao Chen and Zhi Li and Xing Xue and Sinuo Wang and Shaoteng Zhang and Yutong Xie and Yong Xia and Qi Wu and Zhongyi Shui and Xi Li and Zhilin Zheng and Yanjie Zhou and Tony C.W. Mok and Yingda Xia and Hongkan Wang and Xianghua Ye and Tao Ma and Jie Peng and Xiaoguang Wang and Jian Ding and Yuming Gao and Huazhen Ye and Yiping Liu and Dongjie Chen and Zhaomin Ni and Jianwen Ning and Wei Zhang and Jian Liu and Chaohui Yu and Shenghong Ju and Jianfeng Zhang and Wenbo Xiao and Ling Zhang and Tingbo Liang},
    title = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume = {393},
    number = {6817},
    pages = {eaec6129},
    year = {2026},
    doi = {10.1126/science.aec6129}
}
```
