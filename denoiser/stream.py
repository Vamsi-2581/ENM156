"""
Live denoising: feed audio in chunks of stft.hop samples, get denoised
chunks of the same size back. Uses the same causal frames, features and
model as offline processing (the model sees its last `context` input frames,
or carries its state from frame to frame if it is a spiking one), so the
output equals pipeline.enhance on the whole recording, delayed by
n_fft - hop samples.

  stream = StreamingDenoiser(model, cfg, device)
  for chunk in microphone_chunks(cfg["stft"]["hop"]):
      speaker.play(stream.process(chunk))
"""
import torch

from .features import pool_long_mag, window
from .pipeline import input_frames, output_spec
from .snn.nets import SpikingModel


class StreamingDenoiser:
    def __init__(self, model, cfg, device):
        stft = cfg["stft"]
        if not stft["causal"]:
            raise ValueError("live processing needs a model trained with stft.causal = true")
        self.model, self.cfg, self.device = model.eval(), cfg, device
        self.n_fft, self.hop, self.long_factor = stft["n_fft"], stft["hop"], stft["long_factor"]
        self.long_n_fft = self.n_fft * self.long_factor
        self.window = window(self.n_fft, device)
        self.long_window = window(self.long_n_fft, device) if self.long_factor else None
        # The last samples heard; starts as silence, like the zeros analysis()
        # puts before a recording
        self.history = torch.zeros(max(self.n_fft, self.long_n_fft), device=device)
        # Overlap-add of output frames still waiting for later frames, and the
        # summed squared windows they get normalized by
        self.pending = torch.zeros(self.n_fft, device=device)
        self.envelope = torch.zeros(self.n_fft, device=device)
        self.frames = None  # the model's last `context` input frames, from the first chunk on
        self.state = None  # of a spiking model, instead of those frames

    @property
    def latency(self):
        """Samples from an input sample arriving to it leaving process()."""
        return self.n_fft

    @torch.no_grad()
    def process(self, chunk):
        """chunk: hop new input samples (array or tensor) -> hop output samples."""
        chunk = torch.as_tensor(chunk, dtype=torch.float32, device=self.device)
        if chunk.shape != (self.hop,):
            raise ValueError(f"chunks must be {self.hop} samples, got {tuple(chunk.shape)}")
        self.history = torch.cat([self.history[self.hop:], chunk])

        spec = torch.fft.rfft(self.history[-self.n_fft:] * self.window)[:, None]  # (F, 1)
        long = None
        if self.long_factor:
            long_spec = torch.fft.rfft(self.history[-self.long_n_fft:] * self.long_window)[:, None]
            long = pool_long_mag(long_spec, self.long_factor)  # (1, F)
        x = input_frames(spec, long, self.cfg)  # (1, n_in)
        if isinstance(self.model, SpikingModel):
            if self.state is None:  # at rest, as model() starts a clip
                self.state = self.model.init_state((1,), self.device)
            pred, self.state = self.model.step(x, self.state)
        else:
            if self.frames is None:  # silent frames before the start, as model() pads a clip
                self.frames = torch.zeros(self.model.context, x.shape[-1], device=self.device)
            self.frames = torch.cat([self.frames[1:], x])
            # Recomputes the whole context each hop: simple, and exactly like offline
            pred = self.model.forward_valid(self.frames)
        out = output_spec(pred, x, spec, self.cfg)[:, 0]
        frame = torch.fft.irfft(out, n=self.n_fft) * self.window

        self.pending += frame
        self.envelope += self.window ** 2
        # The oldest hop samples get no more frames: they're final
        done = self.pending[:self.hop] / self.envelope[:self.hop].clamp(min=1e-8)
        zeros = torch.zeros(self.hop, device=self.device)
        self.pending = torch.cat([self.pending[self.hop:], zeros])
        self.envelope = torch.cat([self.envelope[self.hop:], zeros])
        return done
