"""FEM élastique transitoire relaxé et adjoint discret à checkpoints."""

import numpy as np
from scipy import sparse

from elastic3d import Plate3D, element_3d
from focal_line import focal_line
from reduced import validate


def common_timestep(c):
    """Borne locale valable pour tout assemblage de fractions positives."""
    dx, dy = c.LENGTH/c.NX, c.WIDTH/c.NY
    bounds = []
    for dz, young, poisson, density in (
        (c.THICKNESS/c.PLATE_LAYERS, c.YOUNG, c.POISSON, c.DENSITY),
        (c.STUB_DZ, c.STUB_YOUNG, c.STUB_POISSON, c.STUB_DENSITY),
    ):
        ke = element_3d(dx, dy, dz, young, poisson)
        mass = density*dx*dy*dz/8
        bounds.append(np.max(np.sum(np.abs(ke), axis=1))/mass)
    return .8*2/np.sqrt(max(bounds))


def simulation_setup(c, plate, dt):
    if c.SOURCE_MODE not in ("top", "A", "S"):
        raise ValueError("SOURCE_MODE doit valoir 'top', 'A' ou 'S'.")
    if not np.isfinite((dt, c.DURATION, c.CYCLES, c.SOURCE_FORCE)).all() or (
        dt <= 0 or dt > plate.dt_limit*(1+1e-12) or c.DURATION <= 0
        or c.CYCLES <= 0 or c.SOURCE_FORCE == 0
        or not 0 <= c.FOCUS_WINDOW[0] < c.FOCUS_WINDOW[1] <= c.DURATION
    ):
        raise ValueError("Paramètres temporels, source ou fenêtre invalides.")
    line_x, y, _, weights = focal_line(c)
    line = np.array([plate.point((line_x, yi), plate.plate_top)[0][2] for yi in y])
    top = plate.point(c.RETURN_SOURCE, plate.plate_top)[0][2]
    bottom = plate.point(c.RETURN_SOURCE, plate.plate_bottom)[0][2]
    times = np.arange(int(np.ceil(c.DURATION/dt))+1)*dt
    window = (times >= c.FOCUS_WINDOW[0]) & (times <= c.FOCUS_WINDOW[1])
    if window.sum() < 2:
        raise ValueError("FOCUS_WINDOW doit contenir au moins deux pas de temps.")
    duration = c.CYCLES/c.FREQUENCY
    burst = np.zeros(len(times))
    mask = times <= duration
    burst[mask] = (c.SOURCE_FORCE*np.sin(2*np.pi*c.FREQUENCY*times[mask])
                   *np.sin(np.pi*times[mask]/duration)**2)
    force = np.zeros(len(plate.mass))
    force[top] = 1 if c.SOURCE_MODE == "top" else .5
    if c.SOURCE_MODE != "top":
        force[bottom] = .5 if c.SOURCE_MODE == "A" else -.5
    return times, window, line, weights, force, burst


def energy(c, plate, dt):
    """Énergie intégrée de u_z² sur la ligne, pour une géométrie exacte."""
    times, window, line, weights, force, burst = simulation_setup(c, plate, dt)
    operator = plate.stiffness.multiply((-dt**2/plate.mass)[:, None]).tocsr()
    forcing = dt**2*force/plate.mass
    previous = np.zeros(len(plate.mass))
    current = previous.copy()
    value = 0.
    for n in range(len(times)):
        if window[n]:
            value += dt*np.dot(weights, current[line]**2)
        if n+1 < len(times):
            previous, current = current, 2*current-previous+operator@current+forcing*burst[n]
    if not np.isfinite(value):
        raise RuntimeError("Réponse élastique non finie.")
    return value


class RelaxedPlate:
    def __init__(self, c):
        validate(c)
        if not (np.isfinite(c.V4_VOID_FRACTION) and 0 < c.V4_VOID_FRACTION < 1):
            raise ValueError("V4_VOID_FRACTION doit être strictement entre 0 et 1.")
        if not isinstance(c.V4_CHECKPOINT_INTERVAL, int) or c.V4_CHECKPOINT_INTERVAL < 1:
            raise ValueError("V4_CHECKPOINT_INTERVAL doit être un entier positif.")
        self.c = c
        self.shape = (c.LENS_Y[1]-c.LENS_Y[0], c.LENS_X[1]-c.LENS_X[0])
        self.full = Plate3D(c, np.full(self.shape, c.MAX_HEIGHT, dtype=int))
        self.dt = common_timestep(c)
        self.times, self.window, self.line, self.weights, self.force, self.burst = simulation_setup(
            c, self.full, self.dt)
        bare = Plate3D(c, np.zeros(self.shape, dtype=int))
        self.reference = energy(c, bare, self.dt)
        if not np.isfinite(self.reference) or self.reference <= np.finfo(float).tiny:
            raise RuntimeError("Référence élastique pondérée nulle ou non finie.")
        dx, dy = c.LENGTH/c.NX, c.WIDTH/c.NY
        self.ke = element_3d(dx, dy, c.STUB_DZ, c.STUB_YOUNG, c.STUB_POISSON)
        self.me = c.STUB_DENSITY*dx*dy*c.STUB_DZ/8
        yy, xx = np.indices(self.shape)
        yy, xx = yy.ravel()+c.LENS_Y[0], xx.ravel()+c.LENS_X[0]
        self.dofs = []
        for z in range(c.MAX_HEIGHT):
            sides = []
            for layer in (self.full.plate_top+z, self.full.plate_bottom-1-z):
                base = layer*self.full.nxy+yy*(c.NX+1)+xx
                nodes = base[:, None]+np.array([0, 1, c.NX+1, c.NX+2,
                                                 self.full.nxy, self.full.nxy+1,
                                                 self.full.nxy+c.NX+1, self.full.nxy+c.NX+2])
                ids = self.full.mapping[(nodes[:, :, None]*3+np.arange(3)).reshape(-1, 24)]
                if np.any(ids < 0):
                    raise RuntimeError("Élément du domaine de conception absent.")
                sides.append(ids)
            self.dofs.append(sides)

    def assemble(self, heights):
        h = np.asarray(heights, dtype=float)
        c = self.c
        if h.shape != self.shape or not np.isfinite(h).all() or np.any(h < 0) or np.any(h > c.MAX_HEIGHT):
            raise ValueError("Hauteurs continues incompatibles.")
        h = h.ravel()
        mass = self.full.mass.copy()
        stiffness = self.full.stiffness.copy()
        for z, sides in enumerate(self.dofs):
            fraction = np.clip(h-z, 0, 1)
            delta = (1-c.V4_VOID_FRACTION)*(fraction-1)
            for ids in sides:
                np.add.at(mass, ids.ravel(), np.repeat(delta*self.me, 24))
                rows = np.broadcast_to(ids[:, :, None], (len(ids), 24, 24)).ravel()
                cols = np.broadcast_to(ids[:, None, :], (len(ids), 24, 24)).ravel()
                values = (delta[:, None, None]*self.ke).ravel()
                stiffness += sparse.coo_matrix((values, (rows, cols)), shape=stiffness.shape).tocsr()
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise RuntimeError("Masse relaxée invalide.")
        stiffness.eliminate_zeros()
        return stiffness, mass

    def forward(self, heights, checkpoints=True):
        stiffness, mass = self.assemble(heights)
        operator = stiffness.multiply((-self.dt**2/mass)[:, None]).tocsr()
        forcing = self.dt**2*self.force/mass
        previous = np.zeros(len(mass))
        current = previous.copy()
        saved = {}
        value = 0.
        interval = self.c.V4_CHECKPOINT_INTERVAL
        for n in range(len(self.times)):
            if checkpoints and n % interval == 0:
                saved[n] = (previous.copy(), current.copy())
            if self.window[n]:
                value += self.dt*np.dot(self.weights, current[self.line]**2)
            if n+1 < len(self.times):
                previous, current = current, 2*current-previous+operator@current+forcing*self.burst[n]
        gain = value/self.reference
        if not np.isfinite(gain):
            raise RuntimeError("Objectif élastique relaxé non fini.")
        return gain, (stiffness, mass, operator, forcing, saved)

    def gradient(self, heights, state):
        stiffness, mass, operator, forcing, saved = state
        dt = self.dt
        h = np.asarray(heights).ravel()
        active = [(h > z) & (h < z+1) for z in range(self.c.MAX_HEIGHT)]
        grad = np.zeros_like(h)
        future = np.zeros(len(mass))
        future2 = future.copy()
        starts = sorted(saved)
        for start in reversed(starts):
            end = min(start+self.c.V4_CHECKPOINT_INTERVAL, len(self.times))
            previous, current = saved[start]
            states = [current]
            for n in range(start, end):
                if n+1 < len(self.times):
                    previous, current = current, 2*current-previous+operator@current+forcing*self.burst[n]
                    states.append(current)
            for n in range(end-1, start-1, -1):
                u = states[n-start]
                if n+1 < len(self.times):
                    next_u = states[n-start+1]
                    prev_u = saved[start][0] if n == start else states[n-start-1]
                    acceleration = (next_u-2*u+prev_u)/dt**2
                    scaled = future/mass
                    for z, sides in enumerate(self.dofs):
                        mask = active[z]
                        if not np.any(mask):
                            continue
                        for ids in sides:
                            ids = ids[mask]
                            local_adj = scaled[ids]
                            grad[mask] -= (1-self.c.V4_VOID_FRACTION)*dt**2*(
                                np.sum((u[ids]@self.ke)*local_adj, axis=1)
                                +self.me*np.sum(local_adj*acceleration[ids], axis=1))
                current_adj = 2*future-future2+operator.T@future
                if self.window[n]:
                    current_adj[self.line] += 2*dt*self.weights*u[self.line]/self.reference
                future2, future = future, current_adj
        if not np.isfinite(grad).all():
            raise RuntimeError("Gradient élastique adjoint non fini.")
        return grad.reshape(self.shape)
