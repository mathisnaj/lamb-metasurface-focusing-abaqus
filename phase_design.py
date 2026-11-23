"""Cellules x-z répétées : transmission A et phase cible de focalisation."""

import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import splu

from reduced import geometry, node, validate
from surrogate import ActiveSurrogate, rankings


def a0_wavenumber(c):
    """Racine A0 sous la vitesse de cisaillement de Rayleigh-Lamb exacte."""
    mu = c.YOUNG/(2*(1+c.POISSON))
    lam = c.YOUNG*c.POISSON/((1+c.POISSON)*(1-2*c.POISSON))
    ct = np.sqrt(mu/c.DENSITY)
    cl = np.sqrt((lam+2*mu)/c.DENSITY)
    omega = 2*np.pi*c.FREQUENCY
    kt, kl = omega/ct, omega/cl
    half = c.THICKNESS/2

    def equation(k):
        p = np.sqrt(k*k-kl*kl)
        q = np.sqrt(k*k-kt*kt)
        return (k*k+q*q)**2*np.tanh(p*half)-4*k*k*p*q*np.tanh(q*half)

    candidates = np.geomspace(kt*(1+1e-8), kt*100, 2000)
    values = np.array([equation(k) for k in candidates])
    changes = np.flatnonzero(values[:-1]*values[1:] < 0)
    if not len(changes):
        raise ValueError("Racine A0 introuvable sous la vitesse de cisaillement pour ces paramètres.")
    return brentq(equation, candidates[changes[0]], candidates[changes[0]+1])


def element(dx, dz, young, poisson, density):
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    constitutive = np.array([[lam+2*mu, lam, 0], [lam, lam+2*mu, 0], [0, 0, mu]])
    signs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    stiffness = np.zeros((8, 8))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            gx = signs[:, 0]*(1+signs[:, 1]*eta)/(2*dx)
            gz = signs[:, 1]*(1+signs[:, 0]*xi)/(2*dz)
            b = np.zeros((3, 8))
            b[0, 0::2], b[1, 1::2] = gx, gz
            b[2, 0::2], b[2, 1::2] = gz, gx
            stiffness += b.T @ constitutive @ b*dx*dz/4
    mass = np.full(8, density*dx*dz/4)
    return stiffness, mass


class Strip:
    """Plaque fixe préassemblée et contributions variables des motifs."""

    def __init__(self, c):
        validate(c)
        self.c = c
        width = c.LENS_X[1]-c.LENS_X[0]
        if not isinstance(c.CELL_PERIOD_X, int) or c.CELL_PERIOD_X < 1 or width % c.CELL_PERIOD_X:
            raise ValueError("CELL_PERIOD_X doit diviser exactement la largeur de LENS_X.")
        if not isinstance(c.BAND_WIDTH_Y, int) or c.BAND_WIDTH_Y < 1 or not isinstance(c.BAND_GAP_Y, int) or c.BAND_GAP_Y < 1:
            raise ValueError("BAND_WIDTH_Y et BAND_GAP_Y doivent être des entiers positifs.")
        if not (0 < c.MIN_TRANSMISSION_A <= 1 and np.isfinite(c.PHASE_TOLERANCE) and c.PHASE_TOLERANCE > 0):
            raise ValueError("Seuil de transmission ou tolérance de phase invalide.")
        if not (isinstance(c.STRIP_PADDING_X, int) and isinstance(c.STRIP_ABSORB_X, int)
                and c.STRIP_PADDING_X >= c.STRIP_ABSORB_X+4 and c.STRIP_ABSORB_X >= 2
                and np.isfinite(c.STRIP_DAMPING) and c.STRIP_DAMPING > 0):
            raise ValueError("Padding ou couche amortie de la coupe invalide.")
        self.dx = c.LENGTH/c.NX
        self.nx = width+2*c.STRIP_PADDING_X
        self.nz = c.PLATE_LAYERS+2*c.MAX_HEIGHT
        self.ndof = 2*(self.nx+1)*(self.nz+1)
        self.plate_z = c.MAX_HEIGHT
        self.plate_top = c.MAX_HEIGHT+c.PLATE_LAYERS
        self.first = c.STRIP_PADDING_X
        self.width = width
        plate_ke, plate_me = element(self.dx, c.THICKNESS/c.PLATE_LAYERS,
                                     c.YOUNG, c.POISSON, c.DENSITY)
        self.stub_ke, self.stub_me = element(self.dx, c.STUB_DZ,
                                           c.STUB_YOUNG, c.STUB_POISSON, c.STUB_DENSITY)
        dofs = np.empty((self.nz, self.nx, 8), dtype=int)
        for z in range(self.nz):
            x = np.arange(self.nx)
            n = z*(self.nx+1)+x
            nodes = n[:, None]+np.array([0, 1, self.nx+1, self.nx+2])
            dofs[z] = (2*nodes[:, :, None]+np.arange(2)).reshape(-1, 8)
        self.dofs = dofs
        self.plate_k, self.plate_mass = self.assemble(
            dofs[self.plate_z:self.plate_top].reshape(-1, 8), plate_ke, plate_me)
        probe_x = c.STRIP_ABSORB_X+(c.STRIP_PADDING_X-c.STRIP_ABSORB_X)//2
        source_x = self.nx-probe_x
        self.src = [self.dof(source_x, self.plate_z, 1), self.dof(source_x, self.plate_top, 1)]
        self.probe = [self.dof(probe_x, self.plate_z, 1), self.dof(probe_x, self.plate_top, 1)]
        self.rhs = np.zeros(self.ndof, dtype=complex)
        self.rhs[self.src] = .5
        xnodes = np.arange(self.nx+1)
        distance = np.minimum(xnodes, self.nx-xnodes)
        damping = c.STRIP_DAMPING*np.maximum(0, (c.STRIP_ABSORB_X-distance)/c.STRIP_ABSORB_X)**2
        self.damping = np.tile(np.repeat(damping, 2), self.nz+1)
        self.cache = {}
        self.reference = self.response(np.zeros(c.CELL_PERIOD_X, dtype=int))
        if abs(self.reference) <= np.finfo(float).tiny:
            raise RuntimeError("Référence de transmission A trop faible.")

    def dof(self, x, z, component):
        return 2*(z*(self.nx+1)+x)+component

    def assemble(self, dofs, ke, me):
        n = len(dofs)
        rows = np.broadcast_to(dofs[:, :, None], (n, 8, 8)).ravel()
        cols = np.broadcast_to(dofs[:, None, :], (n, 8, 8)).ravel()
        vals = np.broadcast_to(ke, (n, 8, 8)).ravel()
        k = sparse.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
        mass = np.bincount(dofs.ravel(), weights=np.tile(me, n), minlength=self.ndof)
        return k, mass

    def response(self, pattern):
        c = self.c
        pattern = np.asarray(pattern)
        if pattern.shape != (c.CELL_PERIOD_X,) or np.any(pattern < 0) or np.any(pattern > c.MAX_HEIGHT) or np.any(pattern != np.round(pattern)):
            raise ValueError("Cellule de hauteurs invalide.")
        repeated = np.tile(pattern.astype(int), self.width//c.CELL_PERIOD_X)
        cells = []
        for ix, height in enumerate(repeated):
            for layer in range(height):
                cells.extend((self.dofs[self.plate_top+layer, self.first+ix],
                              self.dofs[self.plate_z-1-layer, self.first+ix]))
        if cells:
            stub_k, stub_mass = self.assemble(np.asarray(cells), self.stub_ke, self.stub_me)
            k, mass = self.plate_k+stub_k, self.plate_mass+stub_mass
        else:
            k, mass = self.plate_k, self.plate_mass
        active = np.flatnonzero(mass > 0)
        omega = 2*np.pi*c.FREQUENCY
        matrix = (k - sparse.diags(mass*(omega**2+1j*omega*self.damping))).tocsc()
        field = splu(matrix[active][:, active]).solve(self.rhs[active])
        indices = np.searchsorted(active, self.probe)
        value = (field[indices[0]]+field[indices[1]])/2
        if not np.isfinite(value):
            raise RuntimeError("Champ de la coupe non fini.")
        return value

    def transfer(self, pattern):
        key = tuple(int(v) for v in pattern)
        if key not in self.cache:
            self.cache[key] = self.response(pattern)/self.reference
        return self.cache[key]


def wrapped(angle):
    return np.angle(np.exp(1j*angle))


def target_phase(c, y, wavenumber):
    x = (c.LENS_X[0]+c.LENS_X[1])*c.LENGTH/(2*c.NX)
    middle = (c.LENS_Y[0]+c.LENS_Y[1])*c.WIDTH/(2*c.NY)
    _, source = node(c, c.RETURN_SOURCE)
    _, focus = node(c, c.FOCUS)
    def path(y0):
        return (np.hypot(x-source[0], y0-source[1])+
                np.hypot(x-focus[0], y0-focus[1]))
    return wrapped(-wavenumber*(path(y)-path(middle)))


def design_lens(c):
    strip = Strip(c)
    k = a0_wavenumber(c)
    width_y = c.LENS_Y[1]-c.LENS_Y[0]
    stride = c.BAND_WIDTH_Y+c.BAND_GAP_Y
    if width_y % stride:
        raise ValueError("La largeur de LENS_Y doit être un multiple de BAND_WIDTH_Y+BAND_GAP_Y.")
    count = width_y//stride
    if c.SYMMETRY_Y and count % 2 and (width_y-c.BAND_WIDTH_Y) % 2:
        raise ValueError("Nombre impair de bandes incompatible avec la symétrie du maillage y.")
    _, source = node(c, c.RETURN_SOURCE)
    _, focus = node(c, c.FOCUS)
    if c.SYMMETRY_Y and (not np.isclose(source[1], c.WIDTH*(c.LENS_Y[0]+c.LENS_Y[1])/(2*c.NY))
                         or not np.isclose(focus[1], c.WIDTH*(c.LENS_Y[0]+c.LENS_Y[1])/(2*c.NY))):
        raise ValueError("Pour SYMMETRY_Y, la source et le foyer doivent être sur le centre de la lentille.")
    design = np.zeros((width_y, strip.width), dtype=int)
    records = []
    # Symmetric construction places the one-cell gaps between strips, not at outer boundaries.
    starts = [i*stride for i in range(count)]
    if c.SYMMETRY_Y:
        starts = [i*stride if i < count/2 else width_y-c.BAND_WIDTH_Y-(count-1-i)*stride
                  for i in range(count)]
        if count % 2:
            starts[count//2] = (width_y-c.BAND_WIDTH_Y)//2
    unique_count = (count+1)//2 if c.SYMMETRY_Y else count
    targets = [target_phase(c, (c.LENS_Y[0]+starts[i]+c.BAND_WIDTH_Y/2)*c.WIDTH/c.NY, k)
               for i in range(unique_count)]
    surrogate = ActiveSurrogate(c, strip, targets)
    surrogate.run()
    keys = list(strip.cache)
    exact = np.asarray([strip.cache[key] for key in keys])
    for i in range(unique_count):
        start = starts[i]
        y = (c.LENS_Y[0]+start+c.BAND_WIDTH_Y/2)*c.WIDTH/c.NY
        target = targets[i]
        best_index = rankings(exact, target, c)[0]
        best = np.asarray(keys[best_index], dtype=int)
        transfer = exact[best_index]
        repeated = np.tile(best, strip.width//c.CELL_PERIOD_X)
        indices = [i, count-1-i] if c.SYMMETRY_Y and i != count-1-i else [i]
        for j in indices:
            s = starts[j]
            actual_y = (c.LENS_Y[0]+s+c.BAND_WIDTH_Y/2)*c.WIDTH/c.NY
            desired = target_phase(c, actual_y, k)
            design[s:s+c.BAND_WIDTH_Y] = repeated
            records.append({"index": j+1, "y_m": actual_y, "cellule": best.tolist(),
                            "phase_cible_rad": float(desired), "phase_obtenue_rad": float(np.angle(transfer)),
                            "erreur_phase_rad": float(wrapped(np.angle(transfer)-desired)),
                            "transmission_amplitude": float(abs(transfer)),
                            "transmission_admissible": bool(abs(transfer) >= c.MIN_TRANSMISSION_A)})
        print(f"Bande {i+1}/{unique_count} : "
              f"transmission {abs(transfer):.3f}, erreur phase {wrapped(np.angle(transfer)-target):.3f} rad",
              flush=True)
    records.sort(key=lambda r: r["index"])
    geometry(c, design)
    return design, records, k, surrogate
