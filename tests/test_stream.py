"""
Checks that live (streamed) denoising gives the same audio as denoising the
whole recording at once, and that causal frames never look ahead. Runs on an
untrained model and random audio, so no dataset is needed.

  .venv/bin/python tests/test_stream.py
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.config import load_config, merge  # noqa: E402
from denoiser.features import analysis, synthesis  # noqa: E402
from denoiser.model import make_model  # noqa: E402
from denoiser.pipeline import enhance  # noqa: E402
from denoiser.stream import StreamingDenoiser  # noqa: E402

CASES = [  # settings changed from config.toml
    {"stft": {"n_fft": 128, "hop": 80, "long_factor": 10}, "model": {"output": "magnitude"}},
    {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0}, "model": {"output": "magnitude"}},
    {"stft": {"n_fft": 256, "hop": 64, "long_factor": 4}, "model": {"output": "complex_mask"}},
    {"stft": {"n_fft": 128, "hop": 64, "long_factor": 0}, "model": {"type": "tcn", "output": "mask"}},
    {"stft": {"n_fft": 128, "hop": 80, "long_factor": 10}, "model": {"type": "tcn", "output": "magnitude"}},
    {"stft": {"n_fft": 256, "hop": 64, "long_factor": 4}, "model": {"type": "tcn", "output": "complex_mask"}},
    {"stft": {"n_fft": 256, "hop": 64, "long_factor": 0},  # as complex_mask was before mag_input / mask_bound
     "model": {"type": "tcn", "output": "complex_mask", "mag_input": False, "mask_bound": 0}},
]
SMALL_TCN = {"channels": 32, "kernel": 3, "dilations": [1, 2, 4, 8]}  # 31 frames, fast on CPU


def streamed(model, cfg, wav):
    """Run wav through StreamingDenoiser and line the output up with wav."""
    n_fft, hop = cfg["stft"]["n_fft"], cfg["stft"]["hop"]
    delay = n_fft - hop
    n_chunks = -(-(len(wav) + delay) // hop)  # enough zeros after the end to flush it out
    padded = torch.cat([wav, torch.zeros(n_chunks * hop - len(wav))])
    stream = StreamingDenoiser(model, cfg, torch.device("cpu"))
    out = torch.cat([stream.process(c) for c in padded.split(hop)])
    return out[delay:delay + len(wav)]


def main():
    torch.manual_seed(0)
    wav = 0.1 * torch.randn(16000)  # 1 s
    failed = False
    for case in CASES:
        cfg = merge(copy.deepcopy(load_config()), copy.deepcopy(case))
        cfg["stft"]["causal"] = True
        cfg["model"].setdefault("type", "mlp")
        cfg["tcn"] = SMALL_TCN
        stft = cfg["stft"]
        model = make_model(cfg).eval()
        name = (f"n_fft={stft['n_fft']} hop={stft['hop']} long_factor={stft['long_factor']} "
                f"{cfg['model']['type']} {cfg['model']['output']}")

        with torch.no_grad():
            offline = enhance(model, wav, cfg)
        live = streamed(model, cfg, wav)
        err = (live - offline).abs().max().item()

        rec = synthesis(analysis(wav, stft["n_fft"], stft), stft, len(wav))
        rec_err = (rec - wav).abs().max().item()

        # zeroing the input from `cut` on must not change the output before
        # cut - (n_fft - hop): later output samples still overlap frames that
        # end after `cut` (that wait is the latency, not a look ahead)
        cut = 8000
        future_zeroed = wav.clone()
        future_zeroed[cut:] = 0
        with torch.no_grad():
            diff = enhance(model, future_zeroed, cfg) - offline
        causal_err = diff[:cut - (stft["n_fft"] - stft["hop"])].abs().max().item()

        ok = err < 1e-4 and rec_err < 1e-5 and causal_err < 1e-5
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: stream vs offline {err:.1e}, "
              f"reconstruction {rec_err:.1e}, looks ahead {causal_err:.1e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
