"""Élasticité 3D Q1 explicite : plaque et plots de hauteurs variables."""

from itertools import product

import numpy as np
from scipy import sparse

from reduced import geometry, node


SIGNS = np.array(list(product((-1, 1), repeat=3)))[:, ::-1]


def element_3d(dx, dy, dz, young, poisson):
    """Intégration complète sauf cisaillements transverses sélectivement réduits."""
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    np.fill_diagonal(c[:3, :3], lam+2*mu)
    c[3, 3] = mu
    ke = np.zeros((24, 24))

    def gradients(coord):
        grad = np.empty((8, 3))
        for axis, spacing in enumerate((dx, dy, dz)):
            other = [j for j in range(3) if j != axis]
            grad[:, axis] = (SIGNS[:, axis]*(1+SIGNS[:, other[0]]*coord[other[0]])
                             *(1+SIGNS[:, other[1]]*coord[other[1]])/(4*spacing))
        return grad

    for xi in product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3):
        b = np.zeros((6, 24))
        for a, (gx, gy, gz) in enumerate(gradients(xi)):
            b[:, 3*a:3*a+3] = [[gx, 0, 0], [0, gy, 0], [0, 0, gz],
                                [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
        ke += b.T @ c @ b * dx*dy*dz/8
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for component, coord in ((0, (xi, 0, 0)), (1, (0, xi, 0))):
            shear = np.zeros(24)
            for a, (gx, gy, gz) in enumerate(gradients(coord)):
                shear[3*a:3*a+3] = [0, gz, gy] if component == 0 else [gz, 0, gx]
            ke += mu*np.outer(shear, shear)*dx*dy*dz/2
    return ke


class Plate3D:
    def __init__(self, c, design):
        self.c = c
        design = geometry(c, design)
        dx, dy = c.LENGTH/c.NX, c.WIDTH/c.NY
        self.nxy = (c.NX+1)*(c.NY+1)
        self.plate_bottom = c.MAX_HEIGHT
        self.plate_top = c.MAX_HEIGHT+c.PLATE_LAYERS
        total = self.nxy*(c.PLATE_LAYERS+2*c.MAX_HEIGHT+1)
        self.occupied = np.zeros((c.PLATE_LAYERS+2*c.MAX_HEIGHT, c.NY, c.NX), dtype=bool)
        self.occupied[self.plate_bottom:self.plate_top] = True
        for z in range(c.MAX_HEIGHT):
            self.occupied[self.plate_top+z, c.LENS_Y[0]:c.LENS_Y[1],
                          c.LENS_X[0]:c.LENS_X[1]] = design > z
            self.occupied[self.plate_bottom-1-z, c.LENS_Y[0]:c.LENS_Y[1],
                          c.LENS_X[0]:c.LENS_X[1]] = design > z
        full_mass = np.zeros(total*3)
        matrices = []
        for z in range(len(self.occupied)):
            yy, xx = np.nonzero(self.occupied[z])
            if not len(xx):
                continue
            plate = self.plate_bottom <= z < self.plate_top
            dz = c.THICKNESS/c.PLATE_LAYERS if plate else c.STUB_DZ
            young = c.YOUNG if plate else c.STUB_YOUNG
            poisson = c.POISSON if plate else c.STUB_POISSON
            density = c.DENSITY if plate else c.STUB_DENSITY
            local = element_3d(dx, dy, dz, young, poisson)
            base = z*self.nxy + yy*(c.NX+1)+xx
            nodes = base[:, None] + np.array([0, 1, c.NX+1, c.NX+2,
                                               self.nxy, self.nxy+1,
                                               self.nxy+c.NX+1, self.nxy+c.NX+2])
            dofs = (nodes[:, :, None]*3+np.arange(3)).reshape(-1, 24)
            np.add.at(full_mass, dofs.ravel(), density*dx*dy*dz/8)
            count = len(dofs)
            rows = np.broadcast_to(dofs[:, :, None], (count, 24, 24)).ravel()
            cols = np.broadcast_to(dofs[:, None, :], (count, 24, 24)).ravel()
            vals = np.broadcast_to(local, (count, 24, 24)).ravel()
            matrices.append(sparse.coo_matrix((vals, (rows, cols)),
                                             shape=(total*3, total*3)).tocsr())
        self.active = np.flatnonzero(full_mass > 0)
        self.mapping = np.full(total*3, -1, dtype=int)
        self.mapping[self.active] = np.arange(len(self.active))
        self.mass = full_mass[self.active]
        stiffness = sum(matrices)[self.active][:, self.active].tocsr()
        stiffness.eliminate_zeros()
        self.stiffness = stiffness
        bound = np.max(np.asarray(abs(stiffness).sum(axis=1)).ravel()/self.mass)
        self.dt_limit = .8*2/np.sqrt(bound)

    def point(self, position, layer):
        n, actual = node(self.c, position)
        dofs = self.mapping[(layer*self.nxy+n)*3+np.arange(3)]
        if np.any(dofs < 0):
            raise RuntimeError("Nœud de mesure absent de la plaque.")
        return dofs, actual

    def surface_dofs(self):
        n = self.plate_top*self.nxy+np.arange(self.nxy)
        return self.mapping[(3*n[:, None]+np.arange(3))].reshape(self.c.NY+1, self.c.NX+1, 3)

    def simulate(self, dt):
        c = self.c
        if c.SOURCE_MODE not in ("top", "S", "A"):
            raise ValueError("SOURCE_MODE doit valoir 'top', 'S' ou 'A'.")
        if not np.isfinite((dt, c.DURATION, c.CYCLES, c.SOURCE_FORCE)).all() or (
                dt <= 0 or dt > self.dt_limit*(1+1e-12) or c.DURATION <= 0 or
                c.CYCLES <= 0 or c.SOURCE_FORCE == 0 or c.FRAMES < 2):
            raise ValueError("Paramètres temporels ou de source invalides.")
        top, source_position = self.point(c.RETURN_SOURCE, self.plate_top)
        bottom, _ = self.point(c.RETURN_SOURCE, self.plate_bottom)
        sensors = [self.point(p, self.plate_top) for p in c.SENSORS]
        sensor_dofs = np.array([s[0] for s in sensors])
        steps = int(np.ceil(c.DURATION/dt))+1
        times = np.arange(steps)*dt
        frames = np.unique(np.linspace(0, steps-1, min(c.FRAMES, steps), dtype=int))
        snapshots = np.empty((len(frames), c.NY+1, c.NX+1, 3), dtype=np.float32)
        traces = np.zeros((steps, len(sensors), 3))
        duration = c.CYCLES/c.FREQUENCY
        burst = np.zeros(steps)
        pulse = times <= duration
        burst[pulse] = np.sin(2*np.pi*c.FREQUENCY*times[pulse])*np.sin(np.pi*times[pulse]/duration)**2
        operator = self.stiffness.multiply((-dt**2/self.mass)[:, None]).tocsr()
        previous = np.zeros(len(self.mass))
        current = np.zeros_like(previous)
        surface = self.surface_dofs()
        frame = 0
        print(f"3D : {len(self.mass):,} ddl, {steps:,} pas de {dt*1e9:.2f} ns", flush=True)
        for step in range(steps):
            traces[step] = current[sensor_dofs]
            if frame < len(frames) and step == frames[frame]:
                snapshots[frame] = current[surface]
                frame += 1
            if step+1 < steps:
                following = 2*current-previous + operator @ current
                # Total force is the same for top and paired excitations.
                factor = 1.0 if c.SOURCE_MODE == "top" else 0.5
                following[top[2]] += dt**2 * factor*c.SOURCE_FORCE*burst[step]/self.mass[top[2]]
                if c.SOURCE_MODE != "top":
                    sign = 1 if c.SOURCE_MODE == "A" else -1
                    following[bottom[2]] += dt**2 * sign*factor*c.SOURCE_FORCE*burst[step]/self.mass[bottom[2]]
                previous, current = current, following
        if not np.isfinite(snapshots).all() or not np.isfinite(traces).all():
            raise RuntimeError("Propagation instable : déplacements non finis.")
        return times, traces, times[frames], snapshots, source_position, [s[1] for s in sensors]
