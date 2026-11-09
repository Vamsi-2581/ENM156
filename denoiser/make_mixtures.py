"""
Makes fixed-length training and validation mixtures (clean speech + noise)
and writes them as parquet files in the same layout as VoiceBank-DEMAND, so
data.train_dir / data.val_dir can point at them. Settings are in [mix].

Two recipes:

  voicebank    speech: the VoiceBank-DEMAND training clips (its 28 speakers);
               noise: what VoiceBank added to them (noisy - clean), joined
               only with noise of the same type (from VoiceBank's log file)
  vctk_demand  speech: every VCTK speaker except VoiceBank's two test
               speakers; noise: the DEMAND recordings, except the five
               environments of VoiceBank's test set

Both write train-*.parquet and val-*.parquet. Validation speech comes from
mix.val_speakers VCTK speakers that are in neither VoiceBank set (the same
ones for both recipes), and its noise from a share of each noise
(mix.val_noise_share) that training never uses. The VoiceBank test set is
left untouched for testing.

Each clip is mix.seconds long: utterances of one speaker joined, each keeping
a short random piece of its own silence at both ends, and noise scaled to an
SNR drawn uniformly from mix.snr_min..snr_max. VoiceBank's speech keeps its
level; VCTK's is quieter as recorded, so each VCTK utterance is scaled to
mix.vctk_level, VoiceBank's typical level. A mixture is only turned down if
it would clip.

  .venv/bin/python -m denoiser.make_mixtures voicebank /path/to/VoiceBank-DEMAND-16k/mixed_8s
  .venv/bin/python -m denoiser.make_mixtures vctk_demand /path/to/VCTK-DEMAND-16k-8s
"""
import argparse
import io
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from .config import add_config_args, load_config, resolve_path
from .data import load_pairs, split_files

# VoiceBank-DEMAND's test set: its speakers, and the DEMAND environments of its noise
VOICEBANK_TEST_SPEAKERS = {"p232", "p257"}
VOICEBANK_TEST_NOISE = {"TBUS", "SCAFE", "DLIVING", "OOFFICE", "SPSQUARE"}
FRAME = 0.01  # seconds per frame when finding where speech starts and ends
ACTIVE_DB = 40  # frames within this many dB of the loudest count as speech
SHARD = 1000  # clips per parquet file


# ---------------------------------------------------------------- speech


def frame_db(wav, sr):
    """Energy in dB of each FRAME-long piece of wav."""
    n = int(FRAME * sr)
    frames = wav[:len(wav) // n * n].reshape(-1, n)
    return 10 * np.log10((frames ** 2).mean(-1) + 1e-12), n


def trim(wav, sr, rng, cfg):
    """The utterance from its first to its last speech frame, plus a random
    mix.pause_min..pause_max seconds of its own silence at each end."""
    db, n = frame_db(wav, sr)
    active = np.flatnonzero(db > db.max() - ACTIVE_DB)
    start, end = active[0] * n, (active[-1] + 1) * n
    pad = lambda: int(rng.uniform(cfg["pause_min"], cfg["pause_max"]) * sr)
    return wav[max(start - pad(), 0):min(end + pad(), len(wav))]


def speech_clips(utterances, length, sr, rng, cfg):
    """One speaker's (id, wav) utterances, shuffled, trimmed and joined, cut
    into clips of `length` samples. The last unfinished clip is dropped.
    Yields (clean, ids of the utterances in it)."""
    order = list(utterances)
    rng.shuffle(order)
    pieces, ids, filled = [], [], 0
    for utt_id, wav in order:
        piece = trim(wav, sr, rng, cfg)
        while len(piece):
            take = piece[:length - filled]
            pieces.append(take)
            ids.append(utt_id)
            filled += len(take)
            piece = piece[len(take):]
            if filled == length:
                yield np.concatenate(pieces), ids
                pieces, ids, filled = [], [], 0


def active_power(wav, sr):
    """Mean power of the speech frames (as the SNR of the mixture is meant)."""
    db, n = frame_db(wav, sr)
    frames = wav[:len(db) * n].reshape(-1, n)
    return (frames[db > db.max() - ACTIVE_DB] ** 2).mean()


# ---------------------------------------------------------------- noise


class JoinedNoise:
    """Noise pieces grouped by type (VoiceBank's residuals). A clip's noise
    is one type: random pieces of it, each at the same level, joined with
    equal-power crossfades."""

    def __init__(self, pieces_by_type, sr, crossfade):
        self.types = sorted(pieces_by_type)
        self.pieces = {t: [p / np.sqrt((p ** 2).mean() + 1e-12) for p in ps]
                       for t, ps in pieces_by_type.items()}
        self.fade = int(crossfade * sr)

    def __call__(self, length, rng):
        kind = rng.choice(self.types)
        ramp = np.sin(np.linspace(0, np.pi / 2, self.fade)) if self.fade else None
        out = np.zeros(0, dtype=np.float32)
        while len(out) < length:
            piece = self.pieces[kind][rng.randrange(len(self.pieces[kind]))]
            if len(out) and self.fade and len(piece) > self.fade:
                overlap = out[-self.fade:] * ramp[::-1] + piece[:self.fade] * ramp
                out = np.concatenate([out[:-self.fade], overlap, piece[self.fade:]])
            else:
                out = np.concatenate([out, piece])
        return out[:length].astype(np.float32), kind


class RecordedNoise:
    """Long noise recordings (DEMAND). A clip's noise is one continuous
    piece of one recording, from a random start."""

    def __init__(self, recordings):
        self.recordings = recordings  # name -> wav

    def __call__(self, length, rng):
        name = rng.choice(sorted(self.recordings))
        wav = self.recordings[name]
        start = rng.randrange(len(wav) - length + 1)
        return wav[start:start + length], name


# ---------------------------------------------------------------- sources


def voicebank_speakers(log_dir):
    """Speakers of VoiceBank's train and test sets, from its log files."""
    speakers = set()
    for name in ("log_trainset_28spk.txt", "log_testset.txt"):
        for line in (log_dir / name).read_text().split("\n"):
            if line.strip():
                speakers.add(line.split()[0].split("_")[0])
    return speakers


def vctk_speakers(vctk_dir):
    """VCTK speaker -> gender ("F"/"M"), from speaker-info.txt."""
    genders = {}
    for line in (vctk_dir / "speaker-info.txt").read_text().split("\n")[1:]:
        fields = line.split()
        if len(fields) >= 3:
            genders[fields[0] if fields[0].startswith(("p", "s")) else "p" + fields[0]] = fields[2]
    return genders


def pick_val_speakers(vctk_dir, log_dir, count, seed):
    """`count` VCTK speakers in neither VoiceBank set, half of each gender."""
    genders = vctk_speakers(vctk_dir)
    audio_dir = vctk_audio_dir(vctk_dir)
    taken = voicebank_speakers(log_dir)
    free = sorted(s for s in genders if s not in taken and (audio_dir / s).is_dir())
    rng = random.Random(seed)
    picked = []
    for gender, n in (("F", count - count // 2), ("M", count // 2)):
        picked += rng.sample([s for s in free if genders[s] == gender], n)
    return sorted(picked)


def vctk_audio_dir(vctk_dir):
    return vctk_dir / "wav48_silence_trimmed"


def vctk_utterances(vctk_dir, speaker, mic, sr, level):
    """A VCTK speaker's utterances as (id, wav) at sr (VCTK is 48 kHz), each
    scaled so its speech frames are at `level` dBFS."""
    out = []
    for path in sorted((vctk_audio_dir(vctk_dir) / speaker).glob(f"*_{mic}.flac")):
        wav, file_sr = sf.read(path, dtype="float32")
        if file_sr % sr:
            raise SystemExit(f"{path}: {file_sr} Hz is not a multiple of {sr} Hz")
        wav = resample_poly(wav, 1, file_sr // sr).astype(np.float32)
        wav *= 10 ** ((level - 10 * np.log10(active_power(wav, sr) + 1e-12)) / 20)
        out.append((path.stem.removesuffix(f"_{mic}"), wav))
    return out


def voicebank_train(cfg, log_dir):
    """VoiceBank's training clips: speaker -> [(id, clean)], and noise type ->
    [noisy - clean], in the order of the parquet files."""
    noise_types = {}
    for line in (log_dir / "log_trainset_28spk.txt").read_text().split("\n"):
        if line.strip():
            clip_id, kind, _ = line.split()
            noise_types[clip_id] = kind
    files = split_files({"data": {**cfg["data"], "train_dir": ""}}, "train")  # VoiceBank's own
    ids = [i for f in files for i in pq.read_table(f, columns=["id"]).column("id").to_pylist()]
    speech, noise = defaultdict(list), []
    for clip_id, (noisy, clean) in zip(ids, load_pairs(files, len(ids), cfg["data"]["sample_rate"])):
        speech[clip_id.split("_")[0]].append((clip_id, clean))
        noise.append((noise_types[clip_id], noisy - clean))
    return speech, noise


def split_noise_pieces(noise, share, seed):
    """VoiceBank residuals -> (train, val) dicts of type -> pieces; `share`
    of each type's pieces (at random) go to validation only."""
    by_type = defaultdict(list)
    for kind, wav in noise:
        by_type[kind].append(wav)
    rng = random.Random(seed)
    train, val = {}, {}
    for kind, pieces in by_type.items():
        rng.shuffle(pieces)
        n_val = max(1, round(share * len(pieces)))
        val[kind], train[kind] = pieces[:n_val], pieces[n_val:]
    return train, val


def demand_recordings(demand_dir, share, sr):
    """DEMAND environments outside VoiceBank's test set, channel 1 ->
    (train, val) dicts of name -> wav; the last `share` of each recording
    is validation only."""
    train, val = {}, {}
    for env_dir in sorted(p for p in demand_dir.iterdir() if p.is_dir()):
        if env_dir.name in VOICEBANK_TEST_NOISE or not (env_dir / "ch01.wav").exists():
            continue
        wav, file_sr = sf.read(env_dir / "ch01.wav", dtype="float32")
        if file_sr != sr:
            raise SystemExit(f"{env_dir / 'ch01.wav'}: {file_sr} Hz, expected {sr} Hz")
        cut = round(len(wav) * (1 - share))
        train[env_dir.name], val[env_dir.name] = wav[:cut], wav[cut:]
    if not train:
        raise SystemExit(f"No DEMAND environments (*/ch01.wav) in {demand_dir}")
    return train, val


# ---------------------------------------------------------------- writing


def wav_bytes(wav, sr):
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class ParquetWriter:
    """Writes clips in shards of SHARD rows as <split>-NNNNN-of-MMMMM.parquet
    (VoiceBank-DEMAND's naming); the total is only known at the end, so the
    shards get their final names in close()."""

    def __init__(self, out_dir, split, sr):
        self.out_dir, self.split, self.sr = out_dir, split, sr
        self.rows, self.paths, self.count = [], [], 0

    def add(self, clip_id, clean, noisy, **meta):
        self.rows.append({"id": clip_id,
                          "clean": {"bytes": wav_bytes(clean, self.sr), "path": f"{clip_id}.wav"},
                          "noisy": {"bytes": wav_bytes(noisy, self.sr), "path": f"{clip_id}.wav"},
                          **meta})
        self.count += 1
        if len(self.rows) == SHARD:
            self._flush()

    def _flush(self):
        if self.rows:
            path = self.out_dir / f".{self.split}-part-{len(self.paths):05d}.parquet"
            pq.write_table(pa.Table.from_pylist(self.rows), path)
            self.paths.append(path)
            self.rows = []

    def close(self):
        self._flush()
        for i, path in enumerate(self.paths):
            path.rename(self.out_dir / f"{self.split}-{i:05d}-of-{len(self.paths):05d}.parquet")
        return self.count


def mix(clean, noise, snr, sr):
    """noise scaled to `snr` dB below the speech; both turned down together
    if the sum would clip. Returns (clean, noisy, gain applied)."""
    scale = np.sqrt(active_power(clean, sr) / ((noise ** 2).mean() * 10 ** (snr / 10) + 1e-12))
    noisy = clean + scale * noise
    gain = min(1.0, 0.99 / np.abs(noisy).max())
    return clean * gain, noisy * gain, gain


def write_split(writer, speakers, passes, noise, cfg, sr, rng, tag):
    """Mixtures of every speaker's speech, `passes` times over, each pass
    with new joins, noise and SNRs. speakers: name -> utterances, or name ->
    a function that loads them (VCTK, one speaker at a time)."""
    length = round(cfg["seconds"] * sr)
    turned_down = 0
    for speaker in sorted(speakers):
        utterances = speakers[speaker]
        if callable(utterances):
            utterances = utterances()
        for n_pass in range(passes):
            for n, (clean, ids) in enumerate(speech_clips(utterances, length, sr, rng, cfg)):
                noise_wav, kind = noise(length, rng)
                snr = rng.uniform(cfg["snr_min"], cfg["snr_max"])
                clean, noisy, gain = mix(clean, noise_wav, snr, sr)
                turned_down += gain < 1
                writer.add(f"{tag}_{speaker}_{n_pass}_{n:04d}", clean, noisy, speaker=speaker,
                           noise=kind, snr=snr, gain=gain, sources=",".join(dict.fromkeys(ids)))
        print(f"  {writer.split} {speaker}: {writer.count} clips so far", flush=True)
    count = writer.close()
    print(f"{writer.split}: {count} clips of {cfg['seconds']} s ({count * cfg['seconds'] / 3600:.1f} h), "
          f"{turned_down} turned down to avoid clipping")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", choices=("voicebank", "vctk_demand"))
    ap.add_argument("out_dir", help="Folder for the parquet files (must not hold any yet).")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    m, sr = cfg["mix"], cfg["data"]["sample_rate"]
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.glob("*.parquet")):
        raise SystemExit(f"{out_dir} already holds parquet files; remove them or pick another folder")
    out_dir.mkdir(parents=True, exist_ok=True)

    vctk_dir, log_dir = resolve_path(m["vctk_dir"]), resolve_path(m["voicebank_logs"])
    val_speakers = pick_val_speakers(vctk_dir, log_dir, m["val_speakers"], m["seed"])
    print(f"validation speakers (VCTK, in neither VoiceBank set): {', '.join(val_speakers)}")
    load_vctk = lambda s: (lambda: vctk_utterances(vctk_dir, s, m["vctk_mic"], sr, m["vctk_level"]))
    val_speech = {s: load_vctk(s) for s in val_speakers}

    if args.recipe == "voicebank":
        print("Loading VoiceBank's training clips...")
        train_speech, residuals = voicebank_train(cfg, log_dir)
        train_noise, val_noise = split_noise_pieces(residuals, m["val_noise_share"], m["seed"])
        del residuals
        train_noise = JoinedNoise(train_noise, sr, m["crossfade"])
        val_noise = JoinedNoise(val_noise, sr, m["crossfade"])
        passes = m["voicebank_passes"]
    else:
        excluded = VOICEBANK_TEST_SPEAKERS | set(val_speakers)
        names = sorted(p.name for p in vctk_audio_dir(vctk_dir).iterdir()
                       if p.is_dir() and p.name not in excluded)
        train_speech = {s: load_vctk(s) for s in names}
        train_rec, val_rec = demand_recordings(resolve_path(m["demand_dir"]), m["val_noise_share"], sr)
        print(f"DEMAND environments: {', '.join(sorted(train_rec))}")
        train_noise, val_noise = RecordedNoise(train_rec), RecordedNoise(val_rec)
        passes = m["vctk_passes"]
    print(f"train: {len(train_speech)} speakers, {passes} passes; val: {len(val_speakers)} speakers, 1 pass")

    rng = random.Random(m["seed"])
    write_split(ParquetWriter(out_dir, "train", sr), train_speech, passes, train_noise, m, sr, rng, "train")
    write_split(ParquetWriter(out_dir, "val", sr), val_speech, 1, val_noise, m, sr, rng, "val")


if __name__ == "__main__":
    main()
