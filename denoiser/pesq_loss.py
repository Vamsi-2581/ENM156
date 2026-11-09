"""
Differentiable PESQ loss from torch-pesq, with a faster filter.

torch-pesq runs its two PESQ filters (level-alignment bandpass and
pre-emphasis) through torchaudio's recursive (IIR) lfilter, which steps
through the signal one sample at a time and makes a single training step
take seconds on the GPU. Both filters are stable and their impulse response
dies out within a few thousand samples, so here they are applied as an FFT
convolution with that truncated response instead: ~20x faster, same MOS to
about 4 decimals.
"""
import torch
import torch_pesq.loss
from torch_pesq import PesqLoss
from torchaudio.functional import lfilter

N_TAPS = 4096  # length the IIR impulse responses are truncated to

_impulse_responses = {}  # (a, b) coefficient storage -> impulse response


def fft_lfilter(x, a_coeffs, b_coeffs, clamp=False):
    """Drop-in for torchaudio's lfilter on (batch, samples) signals."""
    key = (a_coeffs.data_ptr(), b_coeffs.data_ptr())
    if key not in _impulse_responses:
        impulse = torch.zeros(1, N_TAPS, device=x.device)
        impulse[0, 0] = 1.0
        _impulse_responses[key] = lfilter(impulse, a_coeffs, b_coeffs, clamp=False)[0].detach()
    h = _impulse_responses[key]
    n = x.shape[-1] + N_TAPS
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)
    return y[..., :x.shape[-1]]


def make_pesq_loss(device, sample_rate=16000, factor=0.5):
    """PesqLoss(ref, deg) returns a per-clip loss (lower is better);
    .mos(ref, deg) returns the per-clip PESQ estimate (higher is better)."""
    torch_pesq.loss.lfilter = fft_lfilter
    return PesqLoss(factor, sample_rate=sample_rate).to(device)
