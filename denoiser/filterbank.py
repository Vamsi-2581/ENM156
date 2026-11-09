"""
A learned encoder/decoder that can stand in for the STFT (as in Conv-TasNet):
the encoder is one strided convolution from the waveform to `filters`
channels per frame, the decoder its transposed convolution (a learned
overlap-add). Step 1 of two-step training (train_filterbank.py) learns them
so that an ideal mask in this latent space gives back the clean speech;
a denoiser can then learn to predict that mask, like it does on the STFT.

Framing is causal like the STFT's: frame t covers the `kernel` samples that
end at sample (t + 1) * stride, with zeros before the recording starts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import analysis, synthesis

ARCH_KEYS = ("filters", "kernel", "stride", "relu")  # what a checkpoint stores


def check_filterbank(fb):
    if not 0 < fb["stride"] <= fb["kernel"]:
        raise SystemExit(f"filterbank.stride must be between 1 and kernel ({fb['kernel']}), "
                         f"got {fb['stride']}: frames would leave gaps")


class Filterbank(nn.Module):
    def __init__(self, filters=256, kernel=32, stride=16, relu=True):
        super().__init__()
        self.kernel, self.stride, self.relu = kernel, stride, relu
        self.enc = nn.Conv1d(1, filters, kernel, stride, bias=False)
        self.dec = nn.ConvTranspose1d(filters, 1, kernel, stride, bias=False)

    def encode(self, wav):
        """(B, samples) -> (B, filters, T) latent."""
        # kernel - stride zeros before the start, so the first frame ends at
        # sample `stride`; a kernel of zeros after the end, so the last samples
        # are covered by as many frames as all the others (as features.analysis)
        x = F.pad(wav, (self.kernel - self.stride, self.kernel))
        z = self.enc(x[:, None])
        return F.relu(z) if self.relu else z

    def decode(self, z, length):
        """(B, filters, T) latent -> (B, length) waveform, lined up with encode's input."""
        start = self.kernel - self.stride  # the zeros encode put before the start
        return self.dec(z)[:, 0, start:start + length]

    def oracle(self, noisy, clean):
        """The best a 0..1 mask on this latent can do: each value scaled by
        the share of it that comes from the speech, |E(s)| / (|E(s)| + |E(n)|)."""
        zs, zn = self.encode(clean).abs(), self.encode(noisy - clean).abs()
        return self.decode(zs / (zs + zn + 1e-8) * self.encode(noisy), noisy.shape[-1])


def make_filterbank(fb):
    check_filterbank(fb)
    return Filterbank(**{k: fb[k] for k in ARCH_KEYS})


def save_filterbank(bank, fb, path):
    torch.save({"config": {k: fb[k] for k in ARCH_KEYS}, "state_dict": bank.state_dict()}, path)


def load_filterbank(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    bank = make_filterbank(checkpoint["config"])
    bank.load_state_dict(checkpoint["state_dict"])
    return bank.to(device)


def stft_oracle(noisy, clean, n_fft, hop, form="ratio"):
    """The same ideal mask on a causal STFT, for comparison: "ratio" is
    |S| / (|S| + |N|) like Filterbank.oracle, "clip" is min(|S| / |X|, 1).
    Both keep the noisy phase."""
    stft = {"n_fft": n_fft, "hop": hop, "causal": True}
    X, S = analysis(noisy, n_fft, stft), analysis(clean, n_fft, stft)
    if form == "ratio":
        mask = S.abs() / (S.abs() + (X - S).abs() + 1e-8)
    else:
        mask = (S.abs() / X.abs().clamp(min=1e-8)).clamp(max=1)
    return synthesis(mask * X, stft, noisy.shape[-1])


def snr(est, ref):
    """Signal-to-noise ratio of est against ref in dB, per clip (..., samples)."""
    signal = ref.pow(2).sum(-1)
    error = (ref - est).pow(2).sum(-1)
    return 10 * torch.log10((signal + 1e-8) / (error + 1e-8))


def describe_filterbank(fb, sample_rate):
    ms = 1000 / sample_rate
    return (f"learned filterbank: {fb['filters']} filters of {fb['kernel']} samples "
            f"({fb['kernel'] * ms:.1f} ms), stride {fb['stride']} ({sample_rate // fb['stride']} frames/s), "
            f"{'ReLU' if fb['relu'] else 'linear'} latent")
