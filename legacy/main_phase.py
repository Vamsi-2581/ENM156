"""
Phase-aware version of main.py: the same frame-by-frame MLP speech denoiser,
trained on VoiceBank-DEMAND-16k, but predicting a complex ratio mask instead
of a clean magnitude. The mask is complex-multiplied with the noisy
spectrum, so it scales each bin's magnitude and rotates its phase; the noisy
phase is no longer reused. Spectra are power-law compressed (|X|^0.3, phase
kept) and fed to the model as real and imaginary parts.
"""
import argparse
import ctypes
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_DIR = Path(
    "/home/maren04/Documents/code/LUMINA/datasets/VoiceBank-DEMAND-16k/data"
)
SAMPLE_RATE = 16000 # Hz
N_FFT = 512 # Samples per frame. Tradeoff between either time or frequency resolution
HOP = 256 # How many samples the frame window moves between frames
COMPRESS = 0.3 # Magnitude exponent: squeezes the dynamic range like log1p, keeps phase
MAG_WEIGHT = 0.7 # Loss weight of the magnitude term vs the real/imag term


class TwoLayerNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=(1024, 1024)):
        super().__init__()
        layers = []
        in_dim = n_in
        for out_dim in hidden:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers += [nn.Linear(in_dim, n_out)]  # mask parts can be negative
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_model():
    n_feat = 2 * (N_FFT // 2 + 1)  # real and imaginary part of every bin
    return TwoLayerNN(n_feat, n_feat)


def compress(spec):
    return torch.polar(spec.abs().pow(COMPRESS), spec.angle())


def decompress(spec):
    return torch.polar(spec.abs().pow(1 / COMPRESS), spec.angle())


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


def spectral_loss(est, clean):
    """MSE on the compressed real/imag parts (gets the phase right) mixed with
    MSE on the compressed magnitude (gets the loudness of each bin right)."""
    return ((1 - MAG_WEIGHT) * F.mse_loss(est, clean)
            + MAG_WEIGHT * F.mse_loss(magnitude(est), magnitude(clean)))


def stft_frames(wav, window): # Short Time Fourier Transform
    spec = torch.stft(
        torch.from_numpy(wav), n_fft=N_FFT, hop_length=HOP, window=window,
        return_complex=True,
    )
    return to_frames(compress(spec))  # (T, 2F)


def build_split(parquet_files, max_pairs, window, device, keep_raw=0):
    """Stream decode+STFT one clip at a time so raw audio and per-clip
    frames never pile up beyond what's needed: only `keep_raw` raw
    (noisy, clean) pairs are retained (for later audio reconstruction),
    everything else is discarded as soon as its frames are extracted.
    Returns compressed real/imag (noisy, clean) frame tensors already on `device`."""
    noisy_chunks, clean_chunks, raw_pairs = [], [], []
    count = 0
    for f in parquet_files:
        if count >= max_pairs:
            break
        table = pq.read_table(str(f))
        for i in range(table.num_rows):
            if count >= max_pairs:
                break
            clean = table.column("clean")[i].as_py()
            noisy = table.column("noisy")[i].as_py()
            c, csr = sf.read(io.BytesIO(clean["bytes"]), dtype="float32")
            n, nsr = sf.read(io.BytesIO(noisy["bytes"]), dtype="float32")
            assert csr == SAMPLE_RATE and nsr == SAMPLE_RATE
            if count < keep_raw:
                raw_pairs.append((n, c))
            length = min(len(n), len(c))
            noisy_chunks.append(stft_frames(n[:length], window))
            clean_chunks.append(stft_frames(c[:length], window))
            count += 1
    noisy_frames = stack_and_free(noisy_chunks, device)
    clean_frames = stack_and_free(clean_chunks, device)
    return noisy_frames, clean_frames, raw_pairs


def stack_and_free(chunks, device, trim_every=256):
    """torch.cat equivalent that doesn't hold the chunks and the result at
    the same time: each chunk is copied into a preallocated tensor and freed
    right away. The big allocation is mmap'd, so its pages only become
    resident as they're written, while the freed chunks are handed back to
    the OS every `trim_every` chunks. Peak stays ~1x the data instead of 2x."""
    total = sum(c.shape[0] for c in chunks)
    out = torch.empty((total, chunks[0].shape[1]), dtype=chunks[0].dtype, device=device)
    chunks.reverse()  # pop() from the end keeps the original order
    pos = 0
    while chunks:
        c = chunks.pop()
        out[pos:pos + c.shape[0]] = c
        pos += c.shape[0]
        del c
        if len(chunks) % trim_every == 0:
            release_freed_memory()
    return out


def release_freed_memory():
    """The per-clip chunks freed after build_split are small heap allocations
    that glibc keeps reserved instead of returning to the OS, so RSS stays at
    the loading peak. malloc_trim hands those pages back."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc


def enhance(model, noisy_wav, window):
    """STFT -> complex mask from the model (frame by frame) -> iSTFT on a
    (batch, samples) tensor. Keeps gradients, so it can be trained through."""
    spec = torch.stft(
        noisy_wav, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True,
    )  # (B, F, T)
    noisy = to_frames(compress(spec))  # (B, T, 2F)
    est = apply_mask(model(noisy), noisy)
    return torch.istft(
        decompress(from_frames(est)), n_fft=N_FFT, hop_length=HOP, window=window,
        length=noisy_wav.shape[-1],
    )


def denoise(model, noisy_wav, window, device):
    model.eval()
    with torch.no_grad():
        wav = torch.from_numpy(noisy_wav).to(device)[None]
        return enhance(model, wav, window.to(device))[0].cpu().numpy()


def save_samples(model, raw_pairs, window, device, out_dir):
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for i, (noisy_wav, clean_wav) in enumerate(raw_pairs):
        sample_dir = samples_dir / f"{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        denoised_wav = denoise(model, noisy_wav, window, device)
        sf.write(str(sample_dir / "clean.wav"), clean_wav, SAMPLE_RATE)
        sf.write(str(sample_dir / "noisy.wav"), noisy_wav, SAMPLE_RATE)
        sf.write(str(sample_dir / "denoised.wav"), denoised_wav, SAMPLE_RATE)
    print(f"Saved {len(raw_pairs)} clean/noisy/denoised sample triples to {samples_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-val", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output-dir", default="output_phase")
    ap.add_argument("--num-samples", type=int, default=10,
                     help="Number of clean/noisy/denoised sample triples to save after training.")
    ap.add_argument("--device", default=None,
                     help="torch device (cuda, cpu, cuda:1, ...). Defaults to cuda if available.")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    train_files = sorted(DATASET_DIR.glob("train-*.parquet"))
    test_files = sorted(DATASET_DIR.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise SystemExit(f"No parquet files found in {DATASET_DIR}")

    window = torch.hann_window(N_FFT)
    print(f"Loading + encoding up to {args.max_train} train pairs...")
    tr_noisy, tr_clean, _ = build_split(train_files, args.max_train, window, device)
    print(f"Loading + encoding up to {args.max_val} val pairs "
          f"(keeping {args.num_samples} raw clips for sample output)...")
    va_noisy, va_clean, va_raw_pairs = build_split(
        test_files, args.max_val, window, device, keep_raw=args.num_samples)
    release_freed_memory()
    print(f"  train frames: {tuple(tr_noisy.shape)}   val frames: {tuple(va_noisy.shape)}")

    model = make_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss_fn = nn.MSELoss()  # magnitude-only MSE: see main.py

    n_samples = tr_noisy.shape[0]
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, y = tr_noisy[idx], tr_clean[idx]
            opt.zero_grad()
            est = apply_mask(model(x), x)
            loss = spectral_loss(est, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_loss = spectral_loss(apply_mask(model(va_noisy), va_noisy), va_clean).item()
        model.train()
        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved model to {out_dir / 'model.pt'}")

    save_samples(model, va_raw_pairs, window, device, out_dir)


if __name__ == "__main__":
    main()
