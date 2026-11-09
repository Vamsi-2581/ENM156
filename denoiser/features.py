"""
STFT analysis/synthesis and the spectral features built on it.

Causal framing (stft.causal = true), for live use: frame t covers the
n_fft samples that end at sample (t + 1) * hop, so it only depends on the
past; the time before the recording starts counts as silence (zeros). A
long window ends at the same sample, so it only adds past context. Centred
framing (causal = false) is torch's default: frame t is centred on sample
t * hop and sees half a window into the future.
"""
import torch
import torch.nn.functional as F


def window(n_fft, device):
    return torch.hann_window(n_fft, device=device)


def analysis(wav, n_fft, stft):
    """(..., samples) -> (..., n_fft // 2 + 1, T) complex spectrum with a
    window of n_fft samples (the short n_fft, or a long window) and the
    shared hop. Short and long windows give the same frames T."""
    return frame_spectrum(pad_for_frames(wav, n_fft, stft), n_fft, stft["hop"])


def pad_for_frames(wav, n_fft, stft):
    """What analysis puts around the waveform before cutting frames, so that
    frame t of a window of n_fft samples starts at sample t * hop of the result."""
    if not stft["causal"]:
        # torch.stft's centring: half a window, mirrored, on both sides
        lead_shape, length = wav.shape[:-1], wav.shape[-1]
        padded = F.pad(wav.reshape(-1, 1, length), (n_fft // 2, n_fft // 2), mode="reflect")
        return padded.reshape(*lead_shape, -1)
    # n_fft - hop zeros before the start, so the first frame ends at sample
    # hop; and a short window of zeros after the end, so the last samples
    # are covered by as many frames as all the others
    return F.pad(wav, (front_padding(n_fft, stft), stft["n_fft"]))


def widest_window(stft):
    """The longest window analysis uses: the long one, if any."""
    return stft["n_fft"] * max(stft["long_factor"], 1)


def front_padding(n_fft, stft):
    """Samples pad_for_frames puts before the start for a window of n_fft."""
    return n_fft - stft["hop"] if stft["causal"] else n_fft // 2


def frame_spectrum(padded, n_fft, hop):
    """Spectrum of the n_fft-sample frames starting every hop samples."""
    return torch.stft(padded, n_fft=n_fft, hop_length=hop, window=window(n_fft, padded.device),
                      center=False, return_complex=True)


def synthesis(spec, stft, length):
    """Inverse of analysis for the short window: (..., F, T) -> (..., length)."""
    n_fft, hop = stft["n_fft"], stft["hop"]
    w = window(n_fft, spec.device)
    if not stft["causal"]:
        return torch.istft(spec, n_fft=n_fft, hop_length=hop, window=w, length=length)
    # Weighted overlap-add, done by hand because torch.istft refuses causal
    # framing: the Hann window is 0 at its first sample, so the padded start
    # isn't covered. That part is trimmed off anyway.
    frames = torch.fft.irfft(spec.transpose(-1, -2), n=n_fft) * w  # (..., T, n_fft)
    lead_shape, n_frames = frames.shape[:-2], frames.shape[-2]
    out_len = (n_frames - 1) * hop + n_fft
    y = overlap_add(frames.reshape(-1, n_frames, n_fft), hop, out_len)
    envelope = overlap_add((w ** 2).expand(1, n_frames, n_fft), hop, out_len)
    y = y / envelope.clamp(min=1e-8)
    start = n_fft - hop  # the zeros analysis put before the start
    return y[:, start:start + length].reshape(*lead_shape, length)


def overlap_add(frames, hop, out_len):
    """(N, T, n_fft) frames -> (N, out_len), frame t added at sample t * hop."""
    n_fft = frames.shape[-1]
    return F.fold(frames.transpose(1, 2), output_size=(1, out_len), kernel_size=(1, n_fft),
                  stride=(1, hop)).reshape(frames.shape[0], out_len)


def pool_long_mag(long_spec, long_factor):
    """Long-window spectrum (..., F_long, T) -> (..., T, F) magnitudes on the
    short window's bins. The long_factor times finer bins are pooled back to
    the short ones by summing their power (the energy in that band), and the
    result is scaled by 1 / long_factor because a longer window sums up
    proportionally more signal. That keeps both noise and tones at about the
    level they have in the short frame."""
    power = long_spec.abs().pow(2).transpose(-1, -2)  # (..., T, F_long)
    lead_shape = power.shape[:-1]
    pooled = F.avg_pool1d(
        power.reshape(-1, 1, power.shape[-1]), kernel_size=long_factor,
        stride=long_factor, padding=long_factor // 2, count_include_pad=False,
    )
    band_power = pooled * long_factor  # mean -> sum over the pooled bins
    return band_power.reshape(*lead_shape, -1).sqrt() / long_factor


def compress(spec, exponent):
    """Power-law compression of the magnitude, phase kept."""
    return torch.polar(spec.abs().pow(exponent), spec.angle())


def decompress(spec, exponent):
    return torch.polar(spec.abs().pow(1 / exponent), spec.angle())


def to_frames(spec):
    """(..., F, T) complex -> (..., T, 2F) real: real parts, then imaginary."""
    return torch.cat([spec.real, spec.imag], dim=-2).transpose(-1, -2)


def from_frames(frames):
    """Inverse of to_frames."""
    real, imag = frames.transpose(-1, -2).chunk(2, dim=-2)
    return torch.complex(real.contiguous(), imag.contiguous())


def apply_mask(mask, noisy):
    """Complex multiply of two (..., 2F) real/imag frame tensors."""
    mr, mi = mask.chunk(2, dim=-1)
    xr, xi = noisy.chunk(2, dim=-1)
    return torch.cat([mr * xr - mi * xi, mr * xi + mi * xr], dim=-1)


def magnitude(frames):
    real, imag = frames.chunk(2, dim=-1)
    return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # eps keeps the gradient finite at 0
