"""
Runs train_pesq.py once per candidate (a named set of --set overrides) and
scores each final model.pt on validation clips with evaluate.py: real PESQ,
STOI, ESTOI and the clean pass-through. One run at a time; the scoring of a
run goes on (on one CPU core) while the next one trains. Results are appended
to <out>/results.tsv; candidates that already have a row there are skipped.

  .venv/bin/python sweep.py --out runs/sweep --until 21:00 \
      --shared train.epochs=10 train.train_every=5 \
      --run base_s0 train.seed=0 --run mse0.5 train.seed=0 pesq.mse_weight=0.5
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

COLUMNS = ("pesq", "stoi", "estoi")


def parse_scores(text):
    """evaluate.py's table -> {"denoised": [pesq, stoi, estoi], ...}"""
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in ("noisy", "denoised", "clean->model"):
            rows[parts[0]] = parts[1:]
    return rows


def finish(pending, results):
    """Waits for a scoring process and appends its row to the results."""
    name, sets, proc, log = pending
    text = proc.communicate()[0]
    log.write_text(text)
    rows = parse_scores(text)
    if "denoised" not in rows:
        print(f"{name}: scoring failed, see {log}", flush=True)
        return
    val = [l for l in (log.parent / "log.txt").read_text().splitlines() if l.startswith("epoch")]
    last = val[-1].split("val_PESQ=")[1].split()[0] if val else "nan"
    cells = [name, *rows["denoised"], rows["clean->model"][0], last, " ".join(sets)]
    with results.open("a") as f:
        f.write("\t".join(cells) + "\n")
    print(f"{datetime.now():%H:%M} RESULT {name}: PESQ {cells[1]}  STOI {cells[2]}  ESTOI {cells[3]}  "
          f"clean->model {cells[4]}  (val estimate {last})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--shared", nargs="*", default=[], help="section.key=value overrides of every run")
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar=("NAME", "SET"),
                    help="a candidate: its name, then its own overrides")
    ap.add_argument("--until", default="", help="HH:MM: no new run is started after this time today")
    ap.add_argument("--val-every", type=int, default=14, help="score each Nth validation clip")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.tsv"
    if not results.exists():
        results.write_text("\t".join(("name", *COLUMNS, "clean_pesq", "val_estimate", "settings")) + "\n")
    done = {line.split("\t")[0] for line in results.read_text().splitlines()[1:]}
    pending = None
    for name, *sets in args.run:
        if name in done:
            continue
        if args.until and datetime.now().strftime("%H:%M") >= args.until:
            print(f"{name}: not started, it is past {args.until}", flush=True)
            continue
        run_dir = out / name
        overrides = [x for s in (*args.shared, *sets, f"train.output_dir={run_dir}") for x in ("--set", s)]
        print(f"{datetime.now():%H:%M} START {name}: {' '.join(sets)}", flush=True)
        train = subprocess.run([sys.executable, "train_pesq.py", *overrides],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if pending:
            finish(pending, results)
            pending = None
        if train.returncode or not (run_dir / "model.pt").exists():
            print(f"{name}: training failed, see {run_dir / 'log.txt'}", flush=True)
            continue
        proc = subprocess.Popen([sys.executable, "evaluate.py", "--model", str(run_dir / "model.pt"),
                                 "--split", "val", "--every", str(args.val_every), "--num-clips", "0"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        pending = (name, sets, proc, run_dir / "evaluate_val.txt")
    if pending:
        finish(pending, results)
    print(f"{datetime.now():%H:%M} SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
