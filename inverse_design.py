"""Géométrie neuronale et optimisation par adjoint élastique transitoire."""

import numpy as np
import torch
from torch import nn

from elastic3d import Plate3D
from elastic_adjoint import RelaxedPlate, energy
from reduced import geometry
from surrogate import device_for


def check_parameters(c):
    for name in ("V2_GATE_SHARPNESS",):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f"{name} doit être positif et fini.")
    if not 0 < c.V2_FILL_FRACTION < 1 or not 0 < c.V2_INITIAL_FILL < 1:
        raise ValueError("V2_FILL_FRACTION et V2_INITIAL_FILL doivent être entre 0 et 1.")
    for axis in ("X", "Y"):
        bounds = getattr(c, f"V2_PERIOD_{axis}_RANGE")
        initial = getattr(c, f"V2_INITIAL_PERIOD_{axis}")
        if len(bounds) != 2 or not np.isfinite((*bounds, initial)).all() or not (2 <= bounds[0] < initial < bounds[1]):
            raise ValueError(f"Période {axis} initiale strictement entre ses bornes, avec minimum >= 2.")
    if not isinstance(c.NN_HIDDEN, int) or c.NN_HIDDEN < 1:
        raise ValueError("NN_HIDDEN doit être positif.")


class GeometryNetwork(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.network = nn.Sequential(nn.Linear(6, c.NN_HIDDEN), nn.Tanh(),
                                     nn.Linear(c.NN_HIDDEN, c.NN_HIDDEN), nn.Tanh(),
                                     nn.Linear(c.NN_HIDDEN, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, np.log(c.V2_INITIAL_FILL/(1-c.V2_INITIAL_FILL)))
        def period_parameter(value, limits):
            ratio = (value-limits[0])/(limits[1]-limits[0])
            return nn.Parameter(torch.tensor(np.log(ratio/(1-ratio)), dtype=torch.float64))
        self.period_x = period_parameter(c.V2_INITIAL_PERIOD_X, c.V2_PERIOD_X_RANGE)
        self.period_y = period_parameter(c.V2_INITIAL_PERIOD_Y, c.V2_PERIOD_Y_RANGE)
        self.network.double()

    def periods(self):
        c = self.c
        px = c.V2_PERIOD_X_RANGE[0]+(c.V2_PERIOD_X_RANGE[1]-c.V2_PERIOD_X_RANGE[0])*torch.sigmoid(self.period_x)
        py = c.V2_PERIOD_Y_RANGE[0]+(c.V2_PERIOD_Y_RANGE[1]-c.V2_PERIOD_Y_RANGE[0])*torch.sigmoid(self.period_y)
        return px, py

    def forward(self):
        c = self.c
        ny = c.LENS_Y[1]-c.LENS_Y[0]
        nx = c.LENS_X[1]-c.LENS_X[0]
        yy, xx = torch.meshgrid(torch.arange(ny, dtype=self.period_x.dtype, device=self.period_x.device)+.5,
                                torch.arange(nx, dtype=self.period_x.dtype, device=self.period_x.device)+.5,
                                indexing="ij")
        px, py = self.periods()
        ax = 2*torch.pi*(xx-nx/2)/px
        ay = 2*torch.pi*(yy-ny/2)/py
        features = torch.stack((2*xx/nx-1, 2*yy/ny-1,
                                torch.cos(ax), torch.sin(ax), torch.cos(ay), torch.sin(ay)), dim=-1)
        height = torch.sigmoid(self.network(features).squeeze(-1))
        threshold = np.cos(np.pi*c.V2_FILL_FRACTION)
        gx = torch.sigmoid(c.V2_GATE_SHARPNESS*(torch.cos(ax)-threshold))
        gy = torch.sigmoid(c.V2_GATE_SHARPNESS*(torch.cos(ay)-threshold))
        heights = c.MAX_HEIGHT*height*gx*gy
        if c.SYMMETRY_X:
            heights = (heights+heights.flip(1))/2
        if c.SYMMETRY_Y:
            heights = (heights+heights.flip(0))/2
        return heights


class ElasticGain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heights, plate):
        data = heights.detach().cpu().numpy()
        gain, state = plate.forward(data)
        ctx.heights, ctx.plate, ctx.state = data, plate, state
        ctx.device, ctx.dtype = heights.device, heights.dtype
        return torch.tensor(gain, dtype=heights.dtype, device=heights.device)

    @staticmethod
    def backward(ctx, upstream):
        grad = ctx.plate.gradient(ctx.heights, ctx.state)
        return upstream*torch.as_tensor(grad, dtype=ctx.dtype, device=ctx.device), None


def optimize(c):
    check_parameters(c)
    if not isinstance(c.V4_STEPS, int) or c.V4_STEPS < 1:
        raise ValueError("V4_STEPS doit être un entier positif.")
    if not isinstance(c.V4_DISCRETE_EVERY, int) or c.V4_DISCRETE_EVERY < 1:
        raise ValueError("V4_DISCRETE_EVERY doit être un entier positif.")
    if not np.isfinite(c.V4_LEARNING_RATE) or c.V4_LEARNING_RATE <= 0:
        raise ValueError("V4_LEARNING_RATE doit être positif et fini.")
    if not np.isfinite(c.V4_HEIGHT_PENALTY) or c.V4_HEIGHT_PENALTY < 0:
        raise ValueError("V4_HEIGHT_PENALTY doit être positif et fini.")
    device = device_for(c.NN_DEVICE)
    torch.manual_seed(c.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(c.SEED)
    print("Préparation du FEM élastique et calcul de la référence sur plaque nue...", flush=True)
    plate = RelaxedPlate(c)
    print(f"FEM relaxé : {len(plate.full.mass):,} ddl, {len(plate.times):,} pas de temps, "
          f"dt = {plate.dt*1e9:.2f} ns.", flush=True)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    network = GeometryNetwork(c).to(dtype=dtype).to(device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=c.V4_LEARNING_RATE)
    shape = (c.LENS_Y[1]-c.LENS_Y[0], c.LENS_X[1]-c.LENS_X[0])
    best = np.zeros(shape, dtype=int)
    best_gain, best_score = 1.0, 1.0
    best_periods, best_step, best_state = None, 0, None
    history = []
    for step in range(c.V4_STEPS+1):
        print(f"Étape {step}/{c.V4_STEPS} : propagation relaxée...", flush=True)
        heights = network()
        # FEM and its adjoint stay on CPU, including for MPS.
        heights_cpu = heights.to(device="cpu").to(dtype=torch.float64)
        gain = ElasticGain.apply(heights_cpu, plate)
        score = gain-c.V4_HEIGHT_PENALTY*heights_cpu.mean()/c.MAX_HEIGHT
        if not torch.isfinite(score):
            raise RuntimeError("Objectif différentiable non fini.")
        if step % c.V4_DISCRETE_EVERY == 0 or step == c.V4_STEPS:
            print("Calcul de la géométrie entière...", flush=True)
            discrete = np.rint(heights.detach().cpu().numpy()).astype(int)
            geometry(c, discrete)
            gain_discrete = energy(c, Plate3D(c, discrete), plate.dt)/plate.reference
            objective_discrete = gain_discrete-c.V4_HEIGHT_PENALTY*float(discrete.mean())/c.MAX_HEIGHT
            px, py = (float(p.detach().cpu()) for p in network.periods())
            history.append((step, float(gain.detach().cpu()), float(score.detach().cpu()),
                            gain_discrete, objective_discrete, px, py))
            if objective_discrete > best_score:
                best, best_gain, best_score = discrete.copy(), gain_discrete, objective_discrete
                best_periods, best_step = (px, py), step
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in network.state_dict().items()}
            print(f"Étape {step}/{c.V4_STEPS} : gain relaxé {float(gain):.3f}, "
                  f"gain entier {gain_discrete:.3f}, meilleur {best_gain:.3f}, "
                  f"périodes {px:.2f} × {py:.2f} éléments", flush=True)
        if step < c.V4_STEPS:
            print("Calcul de l'adjoint et mise à jour du réseau...", flush=True)
            optimizer.zero_grad()
            (-score).backward()
            for parameter in network.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Gradient du réseau ou des périodes absent ou non fini.")
            optimizer.step()
    return best, plate, np.asarray(history), best_gain, best_score, best_periods, best_step, str(device), best_state
