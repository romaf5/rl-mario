"""The world model: what the search will unroll instead of the emulator.

    h(4 frames)          -> latent s0            (representation)
    g(s, action)         -> s', and the events that end a segment or a line:
                            goal (the next route level is reached), dead, forced
                            (the input does nothing now)                (dynamics)
    f(s)                 -> prior over the 12 actions, and W            (prediction)

W is the relative value of smbzero/relvalue.py: frames wasted against perfect play from the
root of the search. The root is s0, so W(s0) = 0 by definition and every unrolled node is
scored against it -- which is all PUCT compares.

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
        self.ev = nn.Sequential(nn.Flatten(), nn.Linear(c * 11 * 11, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, s, a):
        """(s, action) -> (next latent, event logits: goal, dead, forced)"""
        plane = F.one_hot(a.long(), N_ACTIONS).float()[:, :, None, None].expand(-1, -1, s.shape[2], s.shape[3])
        h = self.r2(self.r1(self.conv(torch.cat([s, plane], 1))))
        return norm_latent(h), self.ev(h)


class Prediction(nn.Module):
    def __init__(self, c=LATENT_C, hidden=256):
        super().__init__()
        self.fc = nn.Linear(c * 11 * 11, hidden)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.w = nn.Linear(hidden, 1)

    def forward(self, s):
        """s -> (policy logits, W in frames)"""
        h = F.relu(self.fc(s.flatten(1)))
        return self.pi(h), self.w(h).squeeze(1) * 16.0


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
        s2, ev = self.g(s, a)
        pi, w = self.f(s2)
        return s2, ev, pi, w

    def unroll(self, obs, actions):
        """obs (B,4,84,84), actions (B,K) -> latents, event logits, policy logits, W per step."""
        s = self.h(obs)
        lat, evs, pis, ws = [s], [], [], []
        pi, w = self.f(s)
        pis.append(pi); ws.append(w)
        for k in range(actions.shape[1]):
            s, ev = self.g(s, actions[:, k])
            s = scale_grad(s, 0.5)                      # MuZero: halve the gradient along the unroll
            pi, w = self.f(s)
            lat.append(s); evs.append(ev); pis.append(pi); ws.append(w)
        return lat, evs, pis, ws


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
