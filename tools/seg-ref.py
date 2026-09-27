# Per-window segmentation from the PyTorch model on /tmp/radar_ref/image.npy (made by reference.py), for seg-test.mjs.
#   .venv-conv/bin/python tools/seg-ref.py
import sys, numpy as np, torch
sys.argv = ['x']; sys.path.insert(0, __import__('os').path.dirname(__file__))
import reference as ref
from monai.data.utils import dense_patch_slices
img = torch.from_numpy(np.load('/tmp/radar_ref/image.npy'))
while img.dim() < 5: img = img[None]
size = list(img.shape[2:]); print('image', tuple(img.shape))
m = ref.Model(); sd = torch.load(ref.CKPT, map_location='cpu', weights_only=False)['model']
m.load_state_dict(sd, strict=False); m.eval(); torch.set_num_threads(16)
sl = dense_patch_slices(size, ref.ROI, ref.rd._get_scan_interval(size, ref.ROI, 3, ref.OVERLAP))
with torch.inference_mode():
    for i, s in enumerate(sl):
        out = m.visual_encoder(img[[slice(0, 1), slice(None)] + list(s)], None)
        seg_probs, seg = out[0], out[1]
        print(i, [x.start for x in s], 'seg', tuple(seg.shape), seg.dtype, 'probs', tuple(seg_probs.shape))
        np.save(f'/tmp/radar_ref/win{i}.npy', seg.reshape(seg.shape[-3:]).numpy().astype(np.uint8))
        np.save(f'/tmp/radar_ref/win{i}_start.npy', np.array([x.start for x in s]))
