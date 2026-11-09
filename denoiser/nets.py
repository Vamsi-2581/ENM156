"""
The networks. Every one maps input frames (..., T, n_in) to output frames
(..., T, n_out), and output frame t only depends on input frames up to t.

`context` is how many input frames an output frame sees, the current one
included. forward() is for whole clips: it puts context - 1 silent (zero)
frames in front, like the silence before a recording. forward_valid()
instead takes those frames as real history and returns one output per
frame after them; training crops and live streaming use it.
"""
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, None: nn.Identity}


class FrameModel(nn.Module):
    context = 1

    def forward(self, x):
        return self.forward_valid(F.pad(x, (0, 0, self.context - 1, 0)))

    def forward_valid(self, x):
        raise NotImplementedError


class MLP(FrameModel):
    """Each frame on its own (context 1)."""

    def __init__(self, n_in, n_out, hidden=(1024, 1024), out_act="relu"):
        super().__init__()
        layers = []
        in_dim = n_in
        for out_dim in hidden:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        layers += [nn.Linear(in_dim, n_out), ACTIVATIONS[out_act]()]
        self.net = nn.Sequential(*layers)

    def forward_valid(self, x):
        return self.net(x)


class FrameNorm(nn.LayerNorm):
    """LayerNorm over the channels of each frame on its own, for (B, C, T)
    tensors. Normalizing over time would let frames see the future."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """Residual block: a dilated convolution over the last frames, then a
    1x1 convolution that mixes the channels. Unpadded, so the output is
    `trim` frames shorter than the input: those are the history it used."""

    def __init__(self, channels, kernel, dilation):
        super().__init__()
        self.trim = (kernel - 1) * dilation
        self.norm = FrameNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        y = self.mix(F.relu(self.conv(F.relu(self.norm(x)))))
        return x[..., self.trim:] + y


class TCN(FrameModel):
    """Causal dilated convolutions over time, with every input value (each
    frequency bin of each input part) as a channel. With dilations 1, 2, 4,
    ... every block reaches twice as far back as the one before, so the
    context grows exponentially with depth."""

    def __init__(self, n_in, n_out, channels=256, kernel=3, dilations=(1, 2, 4, 8), out_act="relu"):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.Sequential(*(TCNBlock(channels, kernel, d) for d in dilations))
        self.norm = FrameNorm(channels)
        self.out = nn.Conv1d(channels, n_out, 1)
        self.out_act = ACTIVATIONS[out_act]()
        self.context = 1 + sum(block.trim for block in self.blocks)

    def forward_valid(self, x):
        lead_shape, (n_frames, n_in) = x.shape[:-2], x.shape[-2:]
        h = self.blocks(self.inp(x.reshape(-1, n_frames, n_in).transpose(1, 2)))
        y = self.out_act(self.out(F.relu(self.norm(h)))).transpose(1, 2)
        return y.reshape(*lead_shape, *y.shape[1:])
