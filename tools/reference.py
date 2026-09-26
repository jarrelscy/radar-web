"""Reference run of the radar-svc pipeline on one input, dumping intermediates
for checking the browser port: preprocessed volume, window list, which organs
were scored in which window / by the fallback crop, and final scores.

  .venv-conv/bin/python tools/reference.py /tmp/radar_dcm/demo.zip /tmp/radar_ref
"""

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/jarrelscy/radar-svc")
sys.path.insert(0, "/home/jarrelscy/radar-svc/radar")

# inference_demo / server import modules we don't need for the vision path
for name in ("transformers", "pandas", "tqdm"):
    m = types.ModuleType(name)
    m.BertTokenizer = m.tqdm = object
    sys.modules[name] = m
med = types.ModuleType("dynamic_network_architectures.med")
med.XBertEncoder = med.XBertLMHeadDecoder = object
sys.modules["dynamic_network_architectures.med"] = med

import inference_demo as rd  # noqa: E402
import server  # noqa: E402  (radar-svc: _to_nifti and friends)
from monai import transforms  # noqa: E402
from monai.data.utils import dense_patch_slices  # noqa: E402

CKPT = "/data/huggingface/RADAR/checkpoint_radar_pretrain.pth"
ROI, OVERLAP = (96, 256, 384), 0.25


class Model(torch.nn.Module):
    """RADAR minus the text encoder (text features are precomputed)."""

    def __init__(self):
        super().__init__()
        from dynamic_network_architectures.vision_branch import VisionBranch

        self.visual_encoder = VisionBranch()
        self.attention = torch.nn.MultiheadAttention(256, 4, dropout=0.1, batch_first=True)
        self.vision_projs = torch.nn.ModuleList([torch.nn.Linear(256, 256) for _ in range(36)])
        self.query_tokens = torch.nn.Parameter(torch.zeros(36, 256))
        self.temp = torch.nn.Parameter(torch.ones([]))

    forward_test_win = rd.RADAR.forward_test_win.__wrapped__ if hasattr(rd.RADAR.forward_test_win, "__wrapped__") else rd.RADAR.forward_test_win


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    up = work / "upload"
    up.mkdir()
    dst = up / Path(src).name
    dst.symlink_to(Path(src).resolve())
    nii, info = server._to_nifti([dst], work)
    print("nifti", info)

    folder = rd.DataFolder(str(nii.parent))
    image, test_items, meta = folder[0]
    np.save(out / "image.npy", image[0].numpy())
    print("preprocessed", tuple(image.shape))

    model = Model()
    model.organs = folder.organs
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    print(model.load_state_dict(sd, strict=False).unexpected_keys[:2], "...")
    model.eval()
    text_feat = torch.load("/home/jarrelscy/radar-svc/infer_text_embedding_radar.pt", map_location="cpu")
    pad_func = transforms.DivisiblePadd(keys=["image", "label"], k=32, mode="constant", constant_values=0, method="end")

    image = image[None]
    size = list(image.shape[2:])
    slices = dense_patch_slices(size, ROI, rd._get_scan_interval(size, ROI, 3, OVERLAP))
    organ_logits = {k: [] for k in test_items}
    organ_feat, log = {}, {"windows": [], "fallback": []}
    full = torch.zeros((1, 37) + tuple(size))
    count = torch.zeros_like(full)
    with torch.inference_mode():
        for sl in slices:
            win = [slice(0, 1), slice(None)] + list(sl)
            before = set(organ_feat)
            organ_logits, seg_prob = model.forward_test_win(image[win], None, organ_logits, meta["test_organ_names"], text_feat, organ_feat, None)
            full[win] += F.interpolate(seg_prob, size=image[win].shape[2:], mode="trilinear")[0]
            count[win] += 1
            log["windows"].append({"start": [int(s.start) for s in sl], "organs": [o for o in organ_feat if o not in before]})
        stitched = (full / count.clamp(min=1)).argmax(1).unsqueeze(0)
        np.save(out / "stitched.npy", stitched[0, 0].numpy().astype(np.uint8))
        for k, v in organ_logits.items():
            if v:
                continue
            oid = folder.organs.index(k.split("_")[0])
            mask = torch.eq(stitched, oid + 1)
            if not mask.any():
                continue
            box = rd.masks_to_boxes_3d(mask)[0].tolist()
            patch, m = rd.center_crop(image, mask, crop_size=ROI)
            m = m.float()
            m[m == 1] = oid + 1
            pd_ = pad_func({"image": patch[0], "label": m[0]})
            before = set(organ_feat)
            organ_logits, _ = model.forward_test_win(pd_["image"][None], None, organ_logits, meta["test_organ_names"], text_feat, organ_feat, None, skip_organ=oid)
            log["fallback"].append({"item": k, "organ_id": oid, "box": box, "crop_shape": list(pd_["image"].shape[1:]), "organs": [o for o in organ_feat if o not in before]})
    scores = {k: float(np.concatenate(v).mean(0)[1]) for k, v in organ_logits.items() if v}
    log["scores"] = scores
    log["image_shape"] = list(size)
    (out / "reference.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "scores"}, ensure_ascii=False))
    print(len(scores), "scores")


if __name__ == "__main__":
    main(*sys.argv[1:3])
