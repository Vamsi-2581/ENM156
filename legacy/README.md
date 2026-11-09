# Legacy scripts

The original one-file experiments, kept for reference. `train.py` with the
settings below reproduces them (checked: same output to within float
rounding, given the same weights):

| script | train.py settings |
|---|---|
| `main.py` | `stft.long_factor=0 stft.causal=false` (and its `n_fft`/`hop`) |
| `main_memory.py` | `stft.causal=false` |
| `main_phase.py` | `model.output="complex_mask" stft.long_factor=0 stft.causal=false` |

Their checkpoints only store weights; `evaluate.py --model` can still load
them, taking the hop from the config (`--set stft.hop=...`).
