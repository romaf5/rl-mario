"""The world model: what the search will unroll instead of the emulator.

    h(4 frames)          -> latent s0                                  (representation)
    g(s, action)         -> s', the step's waste r, and the events that end a segment or a
                            line: goal, dead, forced (the input does nothing now)  (dynamics)
    f(s)                 -> prior over the 12 actions, and the waste still to come V (prediction)

Waste, not time to go: r is how many frames this step threw away against perfect play
(0 on a perfect line, large when the step commits Mario to a pit), and V is how much more
the position will throw away. A line is scored by the waste it accumulates, sum r + V, which
is what PUCT already compares -- and unlike "frames to the end of the level", both are local
quantities a screen actually shows.

Trained on the agent's own trajectories (frames, actions, events): the latent of an unrolled
step must match the latent of the frames the game really produced (EfficientZero's
consistency loss), and the event heads must fire when the game said so.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import _Res, _Stage

LATENT_C = 32          # latent: LATENT_C x 11 x 11 (the IMPALA trunk's grid)
N_ACTIONS = 12


class Representation(nn.Module):
    def __init__(self, channels=(16, 32, LATENT_C), stack=4):
        super().__init__()
        stages, cin = [], stack
        for c in channels:
            stages.append(_Stage(cin, c)); cin = c
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        return norm_latent(self.stages(x.float() / 255.0))


def norm_latent(s):
    """Scale each latent to [0, 1] (MuZero: keeps the unroll from drifting in scale)."""
    b = s.shape[0]
    lo = s.view(b, -1).min(1)[0].view(b, 1, 1, 1)
    hi = s.view(b, -1).max(1)[0].view(b, 1, 1, 1)
    return (s - lo) / (hi - lo + 1e-5)


class Dynamics(nn.Module):
    def __init__(self, c=LATENT_C):
        super().__init__()
        self.conv = nn.Conv2d(c + N_ACTIONS, c, 3, padding=1)
        self.r1, self.r2 = _Res(c), _Res(c)
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(c * 11 * 11, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, s, a):
        """(s, action) -> (next latent, event logits: goal, dead, forced; the step's waste r >= 0)"""
        plane = F.one_hot(a.long(), N_ACTIONS).float()[:, :, None, None].expand(-1, -1, s.shape[2], s.shape[3])
        h = self.r2(self.r1(self.conv(torch.cat([s, plane], 1))))
        o = self.out(h)
        return norm_latent(h), o[:, :3], F.softplus(o[:, 3]) * 4.0


class Prediction(nn.Module):
    def __init__(self, c=LATENT_C, hidden=256):
        super().__init__()
        self.fc = nn.Linear(c * 11 * 11, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.w = nn.Linear(hidden, 1)

    def forward(self, s):
        """s -> (policy logits, V: the frames this position will still waste, >= 0)"""
        h = F.relu(self.fc(s.flatten(1)))
        return self.pi(h), F.softplus(self.w(h).squeeze(1)) * 16.0


class Projector(nn.Module):
    """SimSiam head for the consistency loss (EfficientZero)."""
    def __init__(self, c=LATENT_C, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(c * 11 * 11, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden))
        self.pred = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, hidden))

    def forward(self, s):
        return self.proj(s)

    def predict(self, s):
        return self.pred(self.proj(s))


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = Representation()
        self.g = Dynamics()
        self.f = Prediction()
        self.proj = Projector()

    def initial(self, obs):
        s = self.h(obs)
        pi, w = self.f(s)
        return s, pi, w

    def step(self, s, a):
        s2, ev, r = self.g(s, a)
        pi, v = self.f(s2)
        return s2, ev, r, pi, v

    def unroll(self, obs, actions):
        """obs (B,4,84,84), actions (B,K) -> latents, event logits, step waste, policy logits, V."""
        s = self.h(obs)
        lat, evs, rs, pis, vs = [s], [], [], [], []
        pi, v = self.f(s)
        pis.append(pi); vs.append(v)
        for k in range(actions.shape[1]):
            s, ev, r = self.g(s, actions[:, k])
            s = scale_grad(s, 0.5)                      # MuZero: halve the gradient along the unroll
            pi, v = self.f(s)
            lat.append(s); evs.append(ev); rs.append(r); pis.append(pi); vs.append(v)
        return lat, evs, rs, pis, vs


def scale_grad(x, k):
    return x * k + x.detach() * (1 - k)


def consistency(model, pred_latents, target_obs):
    """EfficientZero: the unrolled latent must project onto the latent of the real frames."""
    with torch.no_grad():
        t = model.proj(model.h(target_obs))
    p = model.proj.predict(pred_latents)
    return -F.cosine_similarity(p, t.detach(), dim=1).mean()


def save(model, path, **meta):
    torch.save(dict(state=model.state_dict(), **meta), path)


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = WorldModel()
    m.load_state_dict(ck['state'])
    return m.to(device), ck
