"""
Checks that train.py's training crops give the model the same input, target
and output as the whole clip: a crop's frames, made from its stored piece of
audio with history frames in front (zeros before the clip starts), must match
the frames and outputs of the whole clip. Also checks that one epoch's crops
stay inside their clips and don't overlap. Runs on random audio.

  .venv/bin/python tests/test_crops.py
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.data import epoch_crops, stack_waves  # noqa: E402
from denoiser.features import analysis  # noqa: E402
from denoiser.model import make_model  # noqa: E402
from denoiser.pipeline import features, target_frames  # noqa: E402
from train import crop_batch  # noqa: E402

CASES = [  # settings changed from config.toml
    {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0, "causal": True}, "model": {"type": "mlp", "output": "mask"}},
    {"stft": {"n_fft": 128, "hop": 80, "long_factor": 10, "causal": True}, "model": {"type": "tcn", "output": "magnitude"}},
    {"stft": {"n_fft": 256, "hop": 64, "long_factor": 4, "causal": True}, "model": {"type": "tcn", "output": "complex_mask"}},
    {"stft": {"n_fft": 128, "hop": 64, "long_factor": 4, "causal": False}, "model": {"type": "tcn", "output": "mask"}},
]
SMALL_TCN = {"channels": 8, "kernel": 3, "dilations": [1, 2, 4]}


def main():
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    # 16-bit values, as in the WAV files, so storing them as int16 is lossless
    clips = [(torch.randint(-3000, 3000, (n,)) / 32768, torch.randint(-3000, 3000, (n,)) / 32768)
             for n in (3000, 900, 7000, 5000)]
    pairs = [(n.numpy(), c.numpy()) for n, c in clips]
    failed = False

    for case in CASES:
        cfg = merge(copy.deepcopy(load_config()), copy.deepcopy(case))
        cfg["tcn"] = SMALL_TCN
        model = make_model(cfg).eval()
        history = model.context - 1
        waves = stack_waves(pairs, cfg, cpu)
        whole = []  # (input frames, target frames, model output) of each whole clip
        with torch.no_grad():
            for n, c in clips:
                x = features(n, cfg)[1]
                whole.append((x, target_frames(analysis(c, cfg["stft"]["n_fft"], cfg["stft"]), cfg), model(x)))
        lengths_ok = waves.frames.tolist() == [len(x) for x, _, _ in whole]

        for crop in [1, 10, 30]:
            crop_clips, firsts = epoch_crops(waves.frames, crop)
            x, y = crop_batch(waves, crop_clips, firsts, crop, history, cfg, cpu)
            with torch.no_grad():
                pred = model.forward_valid(x)
            err = 0.0
            covered = [torch.zeros(n, dtype=torch.long) for n in waves.frames.tolist()]
            for b, (i, f) in enumerate(zip(crop_clips.tolist(), firsts.tolist())):
                wx, wy, wpred = whole[i]
                ref_x = torch.cat([torch.zeros(max(history - f, 0), wx.shape[1]), wx[max(f - history, 0):f + crop]])
                err = max(err, (x[b] - ref_x).abs().max().item(), (y[b] - wy[f:f + crop]).abs().max().item(),
                          (pred[b] - wpred[f:f + crop]).abs().max().item())
                covered[i][f:f + crop] += 1
                err = err if 0 <= f and f + crop <= len(wx) else float("inf")
            used = sum(int(c.sum()) for c in covered)
            expected = int((waves.frames // crop * crop).sum())
            ok = lengths_ok and err < 1e-4 and max(int(c.max()) for c in covered) <= 1 and used == expected
            failed |= not ok
            stft = cfg["stft"]
            print(f"{'ok  ' if ok else 'FAIL'} n_fft={stft['n_fft']} hop={stft['hop']} "
                  f"long_factor={stft['long_factor']} causal={stft['causal']} {cfg['model']['type']} "
                  f"{cfg['model']['output']} crop={crop}: crop vs whole clip {err:.1e}, "
                  f"{used}/{int(waves.frames.sum())} frames used")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
