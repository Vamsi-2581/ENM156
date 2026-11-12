"""
What the model sees and predicts, for every combination of settings:

  input  = short noisy frame (log1p magnitude, or compressed real/imag for
           complex_mask, then its log1p magnitude if model.mag_input)
           + pooled long-window frame if stft.long_factor > 0
           (with model.filterbank: the compressed latent of that frozen learned
           filterbank instead, and the output is a 0..1 mask on the latent)
  output = magnitude:    clean log1p magnitude (noisy phase reused)
           mask:         gain in 0..1 per bin for the noisy magnitude (noisy
                         phase reused); scored on the log1p magnitude it gives
           complex_mask: a complex mask multiplied with the noisy frame
                         (fixes phase too)
"""
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from .config import resolve_path
from .features import (analysis, apply_mask, compress, decompress, frame_spectrum, from_frames,
                       front_padding, magnitude, pool_long_mag, synthesis, to_frames, widest_window)
from .filterbank import load_filterbank

OUTPUTS = ("magnitude", "mask", "complex_mask")
OUT_ACTIVATIONS = {"magnitude": "relu", "mask": "sigmoid", "complex_mask": None}


_banks = {}  # (path, device) -> loaded filterbank


def filterbank(cfg, device="cpu"):
    """The learned filterbank (train_filterbank.py) that stands in for the
    STFT when model.filterbank is set, else None. It is frozen: only the
    network that predicts the mask on its latent is trained."""
    path = cfg["model"].get("filterbank")
    if not path:
        return None
    if cfg["model"]["output"] != "mask":
        raise SystemExit('model.filterbank needs model.output = "mask"')
    key = (path, str(device))
    if key not in _banks:
        _banks[key] = load_filterbank(resolve_path(path), device).eval().requires_grad_(False)
    return _banks[key]


def hop(cfg):
    """Samples between frames."""
    bank = filterbank(cfg)
    return bank.stride if bank is not None else cfg["stft"]["hop"]


def n_freq(cfg):
    return cfg["stft"]["n_fft"] // 2 + 1


def is_complex(cfg):
    output = cfg["model"]["output"]
    if output not in OUTPUTS:
        raise SystemExit(f"model.output must be one of {OUTPUTS}, got {output!r}")
    return output == "complex_mask"


def out_activation(cfg):
    """What the model's last layer goes through: magnitudes are non-negative,
    a mask lies in 0..1, a complex mask is unbounded."""
    is_complex(cfg)  # checks model.output
    return OUT_ACTIVATIONS[cfg["model"]["output"]]


def check_stft(cfg):
    stft = cfg["stft"]
    if not 0 < stft["hop"] < stft["n_fft"]:
        raise SystemExit(f"stft.hop must be between 0 and n_fft ({stft['n_fft']}), "
                         f"got {stft['hop']}: the Hann windows wouldn't overlap")


def mag_input(cfg):
    return is_complex(cfg) and cfg["model"]["mag_input"]


def io_sizes(cfg):
    bank = filterbank(cfg)
    if bank is not None:
        return bank.enc.out_channels, bank.enc.out_channels
    n_short = n_freq(cfg) * (2 if is_complex(cfg) else 1)
    n_mag = n_freq(cfg) if mag_input(cfg) else 0
    n_long = n_freq(cfg) if cfg["stft"]["long_factor"] else 0
    return n_short + n_mag + n_long, n_short


def long_mag(wav, cfg):
    stft, factor = cfg["stft"], cfg["stft"]["long_factor"]
    return pool_long_mag(analysis(wav, stft["n_fft"] * factor, stft), factor)


def input_frames(spec, long, cfg):
    """spec: short noisy spectrum (..., F, T); long: pooled long-window
    magnitudes (..., T, F), or None without a long window."""
    log_mag = torch.log1p(spec.abs().transpose(-1, -2))
    if is_complex(cfg):
        x = to_frames(compress(spec, cfg["model"]["compress"]))
        if mag_input(cfg):
            x = torch.cat([x, log_mag], dim=-1)
    else:
        x = log_mag
    if long is not None:
        x = torch.cat([x, torch.log1p(long)], dim=-1)
    return x


def target_frames(clean_spec, cfg):
    if is_complex(cfg):
        return to_frames(compress(clean_spec, cfg["model"]["compress"]))
    return torch.log1p(clean_spec.abs().transpose(-1, -2))


def features(wav, cfg):
    """Noisy short spectrum (..., F, T) and model input frames (..., T, n_in)."""
    spec = analysis(wav, cfg["stft"]["n_fft"], cfg["stft"])
    long = long_mag(wav, cfg) if cfg["stft"]["long_factor"] else None
    return spec, input_frames(spec, long, cfg)


def bound_mask(pred, bound):
    """Soft limit on the size of each complex mask value: bound * tanh(|m| /
    bound), which is ~|m| for small values. The direction (the phase shift)
    is kept. bound = 0: no limit."""
    if not bound:
        return pred
    real, imag = pred.chunk(2, dim=-1)
    size = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    scale = bound * torch.tanh(size / bound) / size
    return torch.cat([real * scale, imag * scale], dim=-1)


def masked(pred, x, cfg):
    """complex_mask: the (bounded) mask applied to the noisy short frame in x."""
    return apply_mask(bound_mask(pred, cfg["model"]["mask_bound"]), x[..., :2 * n_freq(cfg)])


def noisy_mag(x, cfg):
    """mask: the noisy short magnitude, back from the log1p in x."""
    return torch.expm1(x[..., :n_freq(cfg)])


def loss_fn(pred, x, y, cfg):
    if cfg["model"]["output"] == "mask":
        return F.mse_loss(torch.log1p(pred * noisy_mag(x, cfg)), y)
    if not is_complex(cfg):
        return F.mse_loss(pred, y)
    # MSE on the compressed real/imag parts (gets the phase right) mixed with
    # MSE on the compressed magnitude (gets the loudness of each bin right)
    est, w = masked(pred, x, cfg), cfg["model"]["mag_weight"]
    return (1 - w) * F.mse_loss(est, y) + w * F.mse_loss(magnitude(est), magnitude(y))


def output_spec(pred, x, spec, cfg):
    """Model output -> denoised short spectrum (..., F, T)."""
    if is_complex(cfg):
        return decompress(from_frames(masked(pred, x, cfg)), cfg["model"]["compress"])
    if cfg["model"]["output"] == "mask":
        return torch.polar(spec.abs() * pred.transpose(-1, -2), spec.angle())
    pred_mag = torch.expm1(pred.transpose(-1, -2)).clamp(min=0)
    return torch.polar(pred_mag, spec.angle())


def enhance(model, noisy_wav, cfg):
    """Noisy (..., samples) -> denoised (..., samples). Keeps gradients, so it
    can be trained through."""
    return run_model(model, noisy_wav, cfg)[0]


def run_model(model, noisy_wav, cfg):
    """enhance(), also returning the model's output and input frames, so a
    frame loss (loss_fn) can be computed from the same forward pass."""
    bank = filterbank(cfg, noisy_wav.device)
    if bank is not None:
        z = bank.encode(noisy_wav)  # (B, filters, T)
        x = (z.sign() * z.abs() ** cfg["model"]["compress"]).transpose(1, 2)
        pred = model(x)
        return bank.decode(pred.transpose(1, 2) * z, noisy_wav.shape[-1]), pred, x
    spec, x = features(noisy_wav, cfg)
    pred = model(x)
    return synthesis(output_spec(pred, x, spec, cfg), cfg["stft"], noisy_wav.shape[-1]), pred, x


def segment_frames(noisy, clean, first, cfg):
    """(input, target) frames, (B, T, n_in) and (B, T, n_out), of waveform
    segments from data.gather_segments: frame t of every window starts at
    sample t * hop of the segment, as in analysis(). first: (B,) the clip's
    frame number of each segment's first frame. Frames before the clip starts
    (negative numbers) are zeros in the input, like the padding a model puts
    in front of a whole clip, so a crop gives the same output as the whole
    clip would."""
    stft = cfg["stft"]
    n_fft, hop, wide = stft["n_fft"], stft["hop"], widest_window(stft)
    n = (noisy.shape[-1] - wide) // hop + 1
    skip = front_padding(wide, stft) - front_padding(n_fft, stft)

    def short(wav):
        return frame_spectrum(wav[..., skip:skip + (n - 1) * hop + n_fft], n_fft, hop)

    spec = short(noisy)
    long = pool_long_mag(frame_spectrum(noisy, wide, hop), stft["long_factor"]) if stft["long_factor"] else None
    x = input_frames(spec, long, cfg)
    in_clip = first.to(x.device)[:, None] + torch.arange(n, device=x.device) >= 0
    return x * in_clip[..., None], target_frames(short(clean), cfg)


def denoise(model, noisy_wav, cfg, device):
    model.eval()
    with torch.no_grad():
        wav = torch.from_numpy(noisy_wav).to(device)[None]
        if filterbank(cfg) is not None:
            # silence up to a whole second: the GPU's convolution library tunes the
            # filterbank's convolutions for every new length, which takes seconds.
            # Everything is causal, so the samples before the padding don't change
            wav = F.pad(wav, (0, -len(noisy_wav) % cfg["data"]["sample_rate"]))
        return enhance(model, wav, cfg)[0, :len(noisy_wav)].cpu().numpy()


def save_samples(model, clips, cfg, device, out_dir):
    """clips: [(name, noisy, clean)], e.g. data.sample_ladder(): one folder
    per clip with the clean, noisy and denoised audio."""
    samples_dir = Path(out_dir) / "samples"
    samples_dir.mkdir(exist_ok=True)
    sr = cfg["data"]["sample_rate"]
    for name, noisy_wav, clean_wav in clips:
        sample_dir = samples_dir / name
        sample_dir.mkdir(exist_ok=True)
        sf.write(str(sample_dir / "clean.wav"), clean_wav, sr)
        sf.write(str(sample_dir / "noisy.wav"), noisy_wav, sr)
        sf.write(str(sample_dir / "denoised.wav"), denoise(model, noisy_wav, cfg, device), sr)
    print(f"Saved {len(clips)} clean/noisy/denoised sample triples to {samples_dir}/")
