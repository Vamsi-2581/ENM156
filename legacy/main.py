"""
Simple 2-layer (one hidden layer) MLP speech denoiser, trained on
VoiceBank-DEMAND-16k. Works frame-by-frame on STFT magnitude spectra:
noisy frame -> clean frame. Noisy phase is reused at inference time.
"""
import argparse
import ctypes
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn

DATASET_DIR = Path(
    "/home/maren04/Documents/code/LUMINA/datasets/VoiceBank-DEMAND-16k/data"
)
SAMPLE_RATE = 16000 # Hz
N_FFT = 512 # Samples per frame. Tradeoff between either time or frequency resolution
HOP = 128 # How many samples the frame window moves between frames


class TwoLayerNN(nn.Module):
    def __init__(self, n_freq, hidden=(1024, 1024)):
        super().__init__()
        layers = []
        in_dim = n_freq
        for out_dim in hidden:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers += [nn.Linear(in_dim, n_freq), nn.ReLU()]  # magnitudes are non-negative
        # layers = layers[1:] # If using nn.Dropout in the loop
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def stft_mag_phase(wav, window): # Short Time Fourier Transform
    spec = torch.stft(
        torch.from_numpy(wav), n_fft=N_FFT, hop_length=HOP, window=window,
        return_complex=True,
    )
    return spec.abs(), spec.angle()


def build_split(parquet_files, max_pairs, window, device, keep_raw=0):
    """Stream decode+STFT one clip at a time so raw audio and per-clip
    frames never pile up beyond what's needed: only `keep_raw` raw
    (noisy, clean) pairs are retained (for later audio reconstruction),
    everything else is discarded as soon as its frames are extracted.
    Returns log1p'd (noisy, clean) frame tensors already on `device`."""
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
            n_mag, _ = stft_mag_phase(n[:length], window)
            c_mag, _ = stft_mag_phase(c[:length], window)
            t = min(n_mag.shape[1], c_mag.shape[1])
            noisy_chunks.append(torch.log1p(n_mag[:, :t].T))  # (T, F)
            clean_chunks.append(torch.log1p(c_mag[:, :t].T))
            count += 1
    noisy_log = stack_and_free(noisy_chunks, device)
    clean_log = stack_and_free(clean_chunks, device)
    return noisy_log, clean_log, raw_pairs


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
    """STFT -> model (frame by frame) -> iSTFT on a (batch, samples) tensor,
    reusing the noisy phase. Keeps gradients, so it can be trained through."""
    spec = torch.stft(
        noisy_wav, n_fft=N_FFT, hop_length=HOP, window=window, return_complex=True,
    )  # (B, F, T)
    pred_log_mag = model(torch.log1p(spec.abs()).transpose(1, 2)).transpose(1, 2)
    pred_mag = torch.expm1(pred_log_mag).clamp(min=0)
    denoised_spec = torch.polar(pred_mag, spec.angle())
    return torch.istft(
        denoised_spec, n_fft=N_FFT, hop_length=HOP, window=window,
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
    ap.add_argument("--output-dir", default="output")
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
    tr_n_log, tr_c_log, _ = build_split(train_files, args.max_train, window, device)
    print(f"Loading + encoding up to {args.max_val} val pairs "
          f"(keeping {args.num_samples} raw clips for sample output)...")
    va_n_log, va_c_log, va_raw_pairs = build_split(
        test_files, args.max_val, window, device, keep_raw=args.num_samples)
    release_freed_memory()
    print(f"  train frames: {tuple(tr_n_log.shape)}   val frames: {tuple(va_n_log.shape)}")

    n_freq = tr_n_log.shape[1]
    model = TwoLayerNN(n_freq).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_samples = tr_n_log.shape[0]
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, y = tr_n_log[idx], tr_c_log[idx]
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_samples

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(va_n_log), va_c_log).item()
        model.train()
        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Saved model to {out_dir / 'model.pt'}")

    save_samples(model, va_raw_pairs, window, device, out_dir)


if __name__ == "__main__":
    main()
