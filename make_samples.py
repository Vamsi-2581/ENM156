"""
Writes a model's sample output again: clean/noisy/denoised triples of the
same validation clips for every model, from no noise to -15 dB SNR (see
data.sample_ladder), into <the model's folder>/samples, so models can be
compared by ear on identical input.

  .venv/bin/python make_samples.py runs/tuned_full/model.pt runs/complex_ft20/best.pt
"""
import argparse
import shutil
from pathlib import Path

from denoiser.config import add_config_args, get_device, load_config
from denoiser.data import sample_ladder
from denoiser.model import load_checkpoint
from denoiser.pipeline import save_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="checkpoints (model.pt, best.pt, ...)")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set)
    device = get_device(cfg)
    clips = sample_ladder(cfg)
    for path in args.models:
        model, model_cfg = load_checkpoint(path, cfg, device)
        out_dir = Path(path).parent
        shutil.rmtree(out_dir / "samples", ignore_errors=True)  # the old ones had other names
        save_samples(model, clips, model_cfg, device, out_dir)


if __name__ == "__main__":
    main()
