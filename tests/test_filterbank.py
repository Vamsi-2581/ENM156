"""
Checks the learned filterbank's framing without training it: that encode +
decode line up with the input, that frames never look ahead, that the
oracle gives the clean clip back when there is no noise, and that a saved
filterbank loads the same. No dataset is needed.

  .venv/bin/python tests/test_filterbank.py
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.filterbank import Filterbank, load_filterbank, save_filterbank  # noqa: E402


def identity_bank(kernel, stride):
    """Filter k passes sample k of the frame through; the decoder adds the
    kernel / stride overlapping frames back up, so decode(encode(x)) == x
    exactly when the framing lines up."""
    bank = Filterbank(filters=kernel, kernel=kernel, stride=stride, relu=False)
    with torch.no_grad():
        bank.enc.weight.copy_(torch.eye(kernel)[:, None, :])
        bank.dec.weight.copy_(torch.eye(kernel)[:, None, :] * stride / kernel)
    return bank


def main():
    torch.manual_seed(0)
    wav = 0.1 * torch.randn(1, 16000)
    checks = {}
    with torch.no_grad():
        for kernel, stride in ((32, 16), (64, 16), (32, 32)):
            bank = identity_bank(kernel, stride)
            out = bank.decode(bank.encode(wav), wav.shape[-1])
            checks[f"reconstruction {kernel}/{stride}"] = (out - wav).abs().max().item()
            out = bank.oracle(wav, wav)  # no noise: the mask is 1 wherever there is signal
            checks[f"oracle without noise {kernel}/{stride}"] = (out - wav).abs().max().item()

        # zeroing the input from `cut` on must not change the output before
        # cut - (kernel - stride), as tests/test_stream.py checks for the STFT
        for relu in (True, False):
            bank = Filterbank(filters=64, kernel=32, stride=16, relu=relu)
            cut = 8000
            future_zeroed = wav.clone()
            future_zeroed[:, cut:] = 0
            full = bank.decode(bank.encode(wav), wav.shape[-1])
            part = bank.decode(bank.encode(future_zeroed), wav.shape[-1])
            checks[f"looks ahead ({'relu' if relu else 'linear'})"] = (
                (full - part)[:, :cut - (32 - 16)].abs().max().item())

        with tempfile.TemporaryDirectory() as tmp:
            fb = {"filters": 64, "kernel": 32, "stride": 16, "relu": True}
            bank = Filterbank(**fb)
            path = Path(tmp) / "filterbank.pt"
            save_filterbank(bank, fb, path)
            loaded = load_filterbank(path, torch.device("cpu"))
            checks["save / load"] = (loaded.oracle(wav, 0.5 * wav) - bank.oracle(wav, 0.5 * wav)).abs().max().item()

    failed = False
    for name, err in checks.items():
        ok = err < 1e-5
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {err:.1e}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
