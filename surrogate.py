"""Apprentissage actif d'une transmission complexe calculée par FEM."""

import copy

import numpy as np
import torch
from torch import nn


def device_for(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise ValueError(f"NN_DEVICE={name!r} indisponible : choisir 'auto', 'cpu', 'cuda' ou 'mps'.")


def validate_options(c):
    for name in ("NN_INITIAL_SAMPLES", "NN_ROUNDS", "NN_EVALUATIONS_PER_ROUND",
                 "NN_POOL_SIZE", "NN_EPOCHS", "NN_HIDDEN"):
        value = getattr(c, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} doit être un entier positif.")
    if c.NN_INITIAL_SAMPLES < 5:
        raise ValueError("NN_INITIAL_SAMPLES doit être supérieur ou égal à 5.")
    if not isinstance(c.NN_RANDOM_PER_ROUND, int) or not 0 <= c.NN_RANDOM_PER_ROUND <= c.NN_EVALUATIONS_PER_ROUND:
        raise ValueError("NN_RANDOM_PER_ROUND doit être compris entre 0 et NN_EVALUATIONS_PER_ROUND.")
    if c.NN_POOL_SIZE < c.NN_EVALUATIONS_PER_ROUND:
        raise ValueError("NN_POOL_SIZE doit être supérieur ou égal à NN_EVALUATIONS_PER_ROUND.")
    if not np.isfinite(c.NN_LEARNING_RATE) or c.NN_LEARNING_RATE <= 0:
        raise ValueError("NN_LEARNING_RATE doit être positif et fini.")
    if not np.isfinite(c.NN_VALIDATION_FRACTION) or not 0 < c.NN_VALIDATION_FRACTION < .5:
        raise ValueError("NN_VALIDATION_FRACTION doit être strictement entre 0 et 0.5.")


def candidates(c, rng, count, existing, parents=()):
    """Génère des cellules distinctes, compatibles avec la symétrie x."""
    seen = set(existing)
    result = []
    independent = (c.CELL_PERIOD_X+1)//2 if c.SYMMETRY_X else c.CELL_PERIOD_X
    available = (c.MAX_HEIGHT+1)**independent - len(seen)
    limit = min(count, max(0, available))
    for _ in range(max(100, 40*limit)):
        if len(result) >= limit:
            break
        if len(parents) and rng.random() < .5:
            pattern = np.array(parents[rng.integers(len(parents))], dtype=int)
            for _ in range(c.MUTATION_PIXELS):
                x = int(rng.integers(independent))
                pattern[x] = int(rng.integers(c.MAX_HEIGHT+1))
                if c.SYMMETRY_X:
                    pattern[-1-x] = pattern[x]
        else:
            pattern = rng.integers(c.MAX_HEIGHT+1, size=c.CELL_PERIOD_X)
            if c.SYMMETRY_X:
                pattern[-independent:] = pattern[:independent][::-1]
        key = tuple(int(v) for v in pattern)
        if key not in seen:
            result.append(pattern)
            seen.add(key)
    return result


def rankings(transfers, target, c):
    amplitudes = np.abs(transfers)
    errors = np.abs(np.angle(np.exp(1j*(np.angle(transfers)-target))))
    feasible = amplitudes >= c.MIN_TRANSMISSION_A
    score = np.where(feasible, amplitudes-errors/c.PHASE_TOLERANCE, amplitudes)
    return np.lexsort((-score, -feasible.astype(int)))


class ActiveSurrogate:
    def __init__(self, c, strip, targets):
        validate_options(c)
        self.c, self.strip = c, strip
        self.targets = np.asarray(targets)
        self.device = device_for(c.NN_DEVICE)
        self.rng = np.random.default_rng(c.SEED)
        torch.manual_seed(c.SEED)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(c.SEED)
        self.diagnostics = []
        self.model = None

    def train(self):
        c = self.c
        keys = list(self.strip.cache)
        x = np.asarray(keys, dtype=np.float32)/c.MAX_HEIGHT
        responses = np.asarray([self.strip.cache[k] for k in keys])
        y = np.column_stack((responses.real, responses.imag)).astype(np.float32)
        if len(x) < 3 or not np.isfinite(y).all():
            raise RuntimeError("Données FEM insuffisantes ou non finies pour entraîner le réseau.")
        order = self.rng.permutation(len(x))
        nval = max(1, min(len(x)-2, int(round(len(x)*c.NN_VALIDATION_FRACTION))))
        val, train = order[:nval], order[nval:]
        xt = torch.tensor(x, device=self.device)
        yt = torch.tensor(y, device=self.device)
        model = nn.Sequential(nn.Linear(c.CELL_PERIOD_X, c.NN_HIDDEN), nn.Tanh(),
                              nn.Linear(c.NN_HIDDEN, c.NN_HIDDEN), nn.Tanh(),
                              nn.Linear(c.NN_HIDDEN, 2)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.NN_LEARNING_RATE)
        best_state, best_loss = None, float("inf")
        for _ in range(c.NN_EPOCHS):
            model.train()
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xt[train]), yt[train])
            if not torch.isfinite(loss):
                raise RuntimeError("Perte d'entraînement du réseau non finie.")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(xt[val]), yt[val]).item()
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        if best_state is None:
            raise RuntimeError("Perte de validation du réseau non finie.")
        model.load_state_dict(best_state)
        self.model = model
        info = {"echantillons_fem": len(x), "entrainement": len(train),
                "validation": len(val), "rmse_validation_composantes": float(np.sqrt(best_loss))}
        self.diagnostics.append(info)
        return info

    def predict(self, patterns):
        inputs = np.asarray(patterns, dtype=np.float32)/self.c.MAX_HEIGHT
        self.model.eval()
        with torch.no_grad():
            values = self.model(torch.tensor(inputs, device=self.device)).cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("Prédictions du réseau non finies.")
        return values[:, 0]+1j*values[:, 1]

    def run(self):
        c = self.c
        for height in range(c.MAX_HEIGHT+1):
            self.strip.transfer(np.full(c.CELL_PERIOD_X, height, dtype=int))
        for pattern in candidates(c, self.rng, max(0, c.NN_INITIAL_SAMPLES-len(self.strip.cache)),
                                  self.strip.cache):
            self.strip.transfer(pattern)
        if len(self.strip.cache) < 3:
            raise ValueError("Au moins trois cellules distinctes sont nécessaires.")
        print(f"Apprentissage initial : {len(self.strip.cache)} cellules FEM ; appareil {self.device}.", flush=True)
        for iteration in range(c.NN_ROUNDS):
            info = self.train()
            keys = list(self.strip.cache)
            exact = np.asarray([self.strip.cache[k] for k in keys])
            parents = [keys[rankings(exact, target, c)[0]] for target in self.targets]
            pool = candidates(c, self.rng, c.NN_POOL_SIZE, self.strip.cache, parents)
            if not pool:
                print("Toutes les cellules accessibles ont été évaluées ou aucune nouvelle cellule n'a pu être générée.", flush=True)
                break
            predictions = self.predict(pool)
            orders = [rankings(predictions, target, c) for target in self.targets]
            selected = []
            seen = set()
            exploit_count = max(0, c.NN_EVALUATIONS_PER_ROUND-c.NN_RANDOM_PER_ROUND)
            # Un candidat par cible avant d'en prendre un deuxième pour la même cible.
            rank = 0
            while len(selected) < min(exploit_count, len(pool)):
                for order in orders:
                    idx = int(order[rank])
                    if idx not in seen:
                        selected.append(idx)
                        seen.add(idx)
                    if len(selected) >= min(exploit_count, len(pool)):
                        break
                rank += 1
            remaining = [i for i in range(len(pool)) if i not in seen]
            self.rng.shuffle(remaining)
            selected.extend(remaining[:c.NN_EVALUATIONS_PER_ROUND-len(selected)])
            for i in selected:
                self.strip.transfer(pool[i])
            print(f"Cycle {iteration+1}/{c.NN_ROUNDS} : {len(selected)} évaluations FEM ; "
                  f"RMSE validation {info['rmse_validation_composantes']:.3g}", flush=True)
        self.train()

    def export(self, output):
        keys = list(self.strip.cache)
        response = np.asarray([self.strip.cache[k] for k in keys])
        np.savez_compressed(output/"donnees_fem.npz",
                            cellules=np.asarray(keys, dtype=int),
                            transmission_reelle=response.real, transmission_imaginaire=response.imag)
        torch.save({"state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "entrees": self.c.CELL_PERIOD_X, "hauteur_max": self.c.MAX_HEIGHT,
                    "neurones_caches": self.c.NN_HIDDEN}, output/"surrogate.pt")
