"""Export RADAR's vision branch to ONNX as three Conv2d-only graphs.

Tensors are laid out [D, C, H, W] with depth as the batch axis, so every op is
2D: a (3,3,3) conv is three shifted Conv2d sums, a stride==kernel transposed
conv is a 1x1 Conv2d + reshape, and concat->conv is split into a sum of convs
(keeps the largest buffer at 1/4-res x 128ch). BatchNorm is folded in.

  part_a  image slab [K+2,1,H,W] + edge mask [K+2]  -> t [K,128,H/4,W/4]
          encoder stages 0-1 per slice, then stage-2 conv0 (valid in D).
  part_b  t for the whole window [D,128,H/4,W/4]    -> x2, emb1, emb2, emb3
          rest of the encoder, decoder stages 0-2, proj1-3.
  part_c  image slab [K,1,H,W] + x2 slab             -> label [K,H,W]
          recomputes encoder stages 0-1, decoder stage 3, seg head, x2
          bilinear upsample, argmax (== VisionBranch pred_mask).

Also writes head.bin/head.json (query tokens, attention, vision_projs, temp,
text embeddings) for the per-organ scoring done in JS.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = "/data/huggingface/RADAR/checkpoint_radar_pretrain.pth"
TEXT = "/home/jarrelscy/radar-svc/infer_text_embedding_radar.pt"
RADAR_SRC = "/tmp/damo-radar/RADAR_inference"
V = "visual_encoder.UNet."


def fold(sd, p):
    """conv + eval BatchNorm at prefix p -> (W, b)."""
    w, b = sd[p + ".conv.weight"], sd[p + ".conv.bias"]
    g, beta = sd[p + ".norm.weight"], sd[p + ".norm.bias"]
    mu, var = sd[p + ".norm.running_mean"], sd[p + ".norm.running_var"]
    s = g / torch.sqrt(var + 1e-5)
    return w * s[:, None, None, None, None], (b - mu) * s + beta


class Conv(nn.Module):
    """Conv3d (+ReLU) on [D,C,H,W]. Several inputs == conv over their concat."""

    def __init__(self, w, b, stride=(1, 1, 1), relu=True, splits=None, valid_d=False):
        super().__init__()
        self.kd, self.kh, self.kw = w.shape[2:]
        self.sd, self.sh, self.sw = stride
        self.relu, self.valid_d = relu, valid_d
        splits = splits or [w.shape[1]]
        ws = torch.split(w, splits, dim=1)
        # per input, per depth tap: [co, ci, kh, kw]
        for i, wi in enumerate(ws):
            for k in range(self.kd):
                self.register_buffer(f"w{i}_{k}", wi[:, :, k].contiguous())
        self.n_in = len(ws)
        self.register_buffer("b", b.contiguous())

    def forward(self, *xs):
        pad = (self.kh // 2, self.kw // 2)
        y = None
        for i, x in enumerate(xs):
            if self.kd == 1:
                terms = [F.conv2d(x, getattr(self, f"w{i}_0"), None, (self.sh, self.sw), pad)]
            else:
                if not self.valid_d:
                    x = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 1))
                n = x.shape[0] - 2  # output depth before D stride
                terms = [
                    F.conv2d(x[k : k + n : self.sd], getattr(self, f"w{i}_{k}"), None, (self.sh, self.sw), pad)
                    for k in range(3)
                ]
            for t in terms:
                y = t if y is None else y + t
        y = y + self.b[None, :, None, None]
        return F.relu(y) if self.relu else y


class Up(nn.Module):
    """ConvTranspose3d with kernel == stride on [D,C,H,W]."""

    def __init__(self, w, b):
        super().__init__()
        ci, co, sd, sh, sw = w.shape
        self.co, self.s = co, (sd, sh, sw)
        self.register_buffer("w", w.permute(1, 2, 3, 4, 0).reshape(co * sd * sh * sw, ci, 1, 1).contiguous())
        self.register_buffer("b", b.repeat_interleave(sd * sh * sw).contiguous())

    def forward(self, x):
        d, _, h, w = x.shape
        sd, sh, sw = self.s
        y = F.conv2d(x, self.w, self.b)  # [D, co*sd*sh*sw, H, W]
        y = y.reshape(d, self.co, sd, sh, sw, h, w).permute(0, 2, 1, 5, 3, 6, 4)
        return y.reshape(d * sd, self.co, h * sh, w * sw)


def enc(sd, s, j, **kw):
    w, b = fold(sd, f"{V}encoder.stages.{s}.0.convs.{j}")
    return w, b


def conv1x1(sd, p):
    w, b = sd[p + ".weight"], sd[p + ".bias"]
    return w, b


class Stem(nn.Module):
    """Encoder stages 0-1 (all (1,3,3) kernels) -> s1 at 1/2 H,W."""

    def __init__(self, sd):
        super().__init__()
        self.c = nn.ModuleList([
            Conv(*enc(sd, 0, 0)),
            Conv(*enc(sd, 0, 1)),
            Conv(*enc(sd, 1, 0), stride=(1, 2, 2)),
            Conv(*enc(sd, 1, 1)),
        ])

    def forward(self, x):
        for c in self.c:
            x = c(x)
        return x


class PartA(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.stem = Stem(sd)
        self.c = Conv(*enc(sd, 2, 0), stride=(1, 2, 2), valid_d=True)

    def forward(self, img, edge):
        s1 = self.stem(img) * edge[:, None, None, None]
        return self.c(s1)


class PartB(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.s2 = Conv(*enc(sd, 2, 1))
        self.s3 = nn.ModuleList([Conv(*enc(sd, 3, 0), stride=(2, 2, 2)), Conv(*enc(sd, 3, 1))])
        self.s4 = nn.ModuleList([Conv(*enc(sd, 4, 0), stride=(2, 2, 2)), Conv(*enc(sd, 4, 1))])
        self.s5 = nn.ModuleList([Conv(*enc(sd, 5, 0), stride=(2, 2, 2)), Conv(*enc(sd, 5, 1))])
        up = lambda i: Up(sd[f"{V}decoder.transpconvs.{i}.weight"], sd[f"{V}decoder.transpconvs.{i}.bias"])
        dec = lambda i, n: Conv(*fold(sd, f"{V}decoder.stages.{i}.convs.0"), splits=[n, n])
        self.u0, self.d0 = up(0), dec(0, 320)
        self.u1, self.d1 = up(1), dec(1, 256)
        self.u2, self.d2 = up(2), dec(2, 128)
        self.p1 = Conv(*conv1x1(sd, "visual_encoder.proj1"), relu=False)
        self.p2 = Conv(*conv1x1(sd, "visual_encoder.proj2"), relu=False)
        self.p3 = Conv(*conv1x1(sd, "visual_encoder.proj3"), relu=False)

    @staticmethod
    def tokens(x):  # [D,C,H,W] -> [D*H*W, C], (d,h,w) order like flatten(2).transpose(1,2)
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])

    def forward(self, t):
        s2 = self.s2(t)
        s3 = self.s3[1](self.s3[0](s2))
        s4 = self.s4[1](self.s4[0](s3))
        s5 = self.s5[1](self.s5[0](s4))
        x = self.d0(self.u0(s5), s4)
        x = self.d1(self.u1(x), s3)
        x2 = self.d2(self.u2(x), s2)
        return x2, self.tokens(self.p1(s5)), self.tokens(self.p2(s4)), self.tokens(self.p3(s3))


class PartC(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.stem = Stem(sd)
        self.u3 = Up(sd[f"{V}decoder.transpconvs.3.weight"], sd[f"{V}decoder.transpconvs.3.bias"])
        self.d3 = Conv(*fold(sd, f"{V}decoder.stages.3.convs.0"), splits=[64, 64])
        self.seg = Conv(*conv1x1(sd, f"{V}decoder.seg_layers.3"), relu=False)

    def forward(self, img, x2):
        s1 = self.stem(img)
        logits = self.seg(self.d3(self.u3(x2), s1))
        up = F.interpolate(logits, scale_factor=2.0, mode="bilinear", align_corners=False)
        return up.argmax(1).to(torch.int32)


def reference(sd):
    sys.path.insert(0, RADAR_SRC)
    from dynamic_network_architectures.vision_branch import VisionBranch

    m = VisionBranch()
    m.load_state_dict({k[len("visual_encoder."):]: v for k, v in sd.items() if k.startswith("visual_encoder.")})
    return m.eval()


def run_split(a, b, c, img, k=8):
    """Drive the three parts over one window like the browser does. img [D,1,H,W]."""
    d = img.shape[0]
    t = []
    for z in range(0, d, k):
        z1 = min(z + k, d)
        lo, hi = z - 1, z1 + 1
        idx = torch.arange(lo, hi).clamp(0, d - 1)
        edge = ((torch.arange(lo, hi) >= 0) & (torch.arange(lo, hi) < d)).float()
        t.append(a(img[idx], edge))
    x2, e1, e2, e3 = b(torch.cat(t))
    lab = torch.cat([c(img[z : z + k], x2[z : z + k]) for z in range(0, d, k)])
    return lab, e1, e2, e3


def verify(sd, a, b, c, shape):
    torch.manual_seed(0)
    ref = reference(sd)
    x = torch.rand(1, 1, *shape)
    with torch.inference_mode():
        _, pred, r1, r2, r3, *_ = ref(x, None)
        lab, e1, e2, e3 = run_split(a, b, c, x[0].permute(1, 0, 2, 3))
    agree = (lab == pred[0]).float().mean().item()
    err = [((u - v[0]).abs().max() / v.abs().max()).item() for u, v in ((e1, r1), (e2, r2), (e3, r3))]
    print(f"verify {shape}: label agreement {agree:.6f}, emb rel max err {['%.2e' % e for e in err]}")
    return agree, err


def export(a, b, c, out: Path, opset=17):
    out.mkdir(parents=True, exist_ok=True)
    K, H, W = 8, 256, 384
    dyn = {0: "k", 2: "h", 3: "w"}
    kw = dict(opset_version=opset, dynamo=False, do_constant_folding=True)
    with torch.inference_mode():
        torch.onnx.export(
            a, (torch.rand(K + 2, 1, H, W), torch.ones(K + 2)), out / "part_a.onnx",
            input_names=["img", "edge"], output_names=["t"],
            dynamic_axes={"img": {0: "k2", 2: "h", 3: "w"}, "edge": {0: "k2"}, "t": {0: "k", 2: "h4", 3: "w4"}}, **kw,
        )
        torch.onnx.export(
            b, (torch.rand(96, 128, H // 4, W // 4),), out / "part_b.onnx",
            input_names=["t"], output_names=["x2", "emb1", "emb2", "emb3"],
            dynamic_axes={"t": {0: "d", 2: "h4", 3: "w4"}, "x2": {0: "d", 2: "h4", 3: "w4"},
                          "emb1": {0: "l1"}, "emb2": {0: "l2"}, "emb3": {0: "l3"}}, **kw,
        )
        torch.onnx.export(
            c, (torch.rand(K, 1, H, W), torch.rand(K, 128, H // 4, W // 4)), out / "part_c.onnx",
            input_names=["img", "x2"], output_names=["label"],
            dynamic_axes={"img": dyn, "x2": {0: "k", 2: "h4", 3: "w4"}, "label": {0: "k", 1: "h", 2: "w"}}, **kw,
        )
    print("part_b external data:", split_external(out / "part_b.onnx"))
    for p in ("part_a", "part_b", "part_c"):
        print(p, f"{(out / (p + '.onnx')).stat().st_size / 1e6:.1f} MB")


def export_head(sd, out: Path):
    text = torch.load(TEXT, map_location="cpu")
    tensors = {
        "query_tokens": sd["query_tokens"],
        "in_proj_weight": sd["attention.in_proj_weight"],
        "in_proj_bias": sd["attention.in_proj_bias"],
        "out_proj_weight": sd["attention.out_proj.weight"],
        "out_proj_bias": sd["attention.out_proj.bias"],
        "vision_projs_weight": torch.stack([sd[f"vision_projs.{i}.weight"] for i in range(36)]),
        "vision_projs_bias": torch.stack([sd[f"vision_projs.{i}.bias"] for i in range(36)]),
        "text_feat": torch.stack(list(text.values())),
    }
    meta, blobs, off = {"temp": float(sd["temp"]), "text_keys": list(text.keys()), "tensors": {}}, [], 0
    for name, t in tensors.items():
        a = t.detach().float().contiguous().numpy()
        meta["tensors"][name] = {"offset": off, "shape": list(a.shape)}
        blobs.append(a.tobytes())
        off += a.nbytes
    (out / "head.bin").write_bytes(b"".join(blobs))
    (out / "head.json").write_text(json.dumps(meta, ensure_ascii=False))
    print("head.bin", f"{off / 1e6:.1f} MB,", len(text), "text keys")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    a, b, c = PartA(sd).eval(), PartB(sd).eval(), PartC(sd).eval()
    if not args.no_verify:
        verify(sd, a, b, c, (96, 256, 384))
    out = Path(args.out)
    export(a, b, c, out)
    export_head(sd, out)


def split_external(path: Path, max_bytes=60_000_000):
    """Move initializers to <name>.data0, .data1, ... each under max_bytes (GitHub Pages caps files at 100MB)."""
    import onnx
    from onnx.external_data_helper import set_external_data

    m = onnx.load(str(path))
    files, cur, idx = {}, 0, 0
    for t in m.graph.initializer:
        raw = t.raw_data
        if len(raw) < 1024:
            continue
        if cur + len(raw) > max_bytes and cur:
            idx, cur = idx + 1, 0
        name = f"{path.stem}.data{idx}"
        f = files.setdefault(name, bytearray())
        set_external_data(t, location=name, offset=len(f), length=len(raw))
        f += raw
        t.ClearField("raw_data")
        t.data_location = onnx.TensorProto.EXTERNAL
        cur += len(raw)
    for name, data in files.items():
        (path.parent / name).write_bytes(data)
    onnx.save(m, str(path))
    return list(files)


if __name__ == "__main__":
    main()
