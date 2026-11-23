"""FEM scalaire A0 en plan : factorisation SciPy et adjoint exact PyTorch."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import torch

from phase_design import a0_wavenumber
from focal_line import focal_line
from reduced import node, validate


def element_q1(dx, dy):
    signs = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    stiffness, mass = np.zeros((4, 4)), np.zeros((4, 4))
    for xi in (-1/np.sqrt(3), 1/np.sqrt(3)):
        for eta in (-1/np.sqrt(3), 1/np.sqrt(3)):
            shape = (1+signs[:, 0]*xi)*(1+signs[:, 1]*eta)/4
            gx = signs[:, 0]*(1+signs[:, 1]*eta)/(2*dx)
            gy = signs[:, 1]*(1+signs[:, 0]*xi)/(2*dy)
            stiffness += (np.outer(gx, gx)+np.outer(gy, gy))*dx*dy/4
            mass += np.outer(shape, shape)*dx*dy/4
    return stiffness, mass


class Plate2D:
    def __init__(self, c):
        validate(c)
        self.c = c
        self.dx, self.dy = c.LENGTH/c.NX, c.WIDTH/c.NY
        self.size = (c.NX+1)*(c.NY+1)
        self.k0 = a0_wavenumber(c)
        self.ke, self.me = element_q1(self.dx, self.dy)
        yy, xx = np.indices((c.NY, c.NX))
        bases = (yy*(c.NX+1)+xx).ravel()
        dofs = bases[:, None]+np.array([0, 1, c.NX+1, c.NX+2])
        selected = np.arange(c.NX*c.NY).reshape(c.NY, c.NX)[
            c.LENS_Y[0]:c.LENS_Y[1], c.LENS_X[0]:c.LENS_X[1]].ravel()
        self.design_dofs = dofs[selected]
        self.rows = np.broadcast_to(self.design_dofs[:, :, None],
                                    (len(selected), 4, 4)).ravel()
        self.cols = np.broadcast_to(self.design_dofs[:, None, :],
                                    (len(selected), 4, 4)).ravel()
        baseline = self.assemble(dofs, self.ke-self.k0**2*self.me).astype(complex)
        edge = sparse.lil_matrix((self.size, self.size), dtype=complex)
        edge_element = np.array([[2, 1], [1, 2]])/6
        for y in range(c.NY):
            for x in (0, c.NX):
                ids = [y*(c.NX+1)+x, (y+1)*(c.NX+1)+x]
                edge[np.ix_(ids, ids)] += -1j*self.k0*self.dy*edge_element
        for x in range(c.NX):
            for y in (0, c.NY):
                ids = [y*(c.NX+1)+x, y*(c.NX+1)+x+1]
                edge[np.ix_(ids, ids)] += -1j*self.k0*self.dx*edge_element
        self.fixed = (baseline+edge.tocsr()).tocsc()
        self.source, self.source_position = node(c, c.RETURN_SOURCE)
        self.focus, self.focus_position = node(c, c.FOCUS)
        self.line_x, self.line_y, self.line_dofs, self.line_weights = focal_line(c)
        self.rhs = np.zeros(self.size, dtype=np.complex128)
        self.rhs[self.source] = 1.0
        self.bare = splu(self.fixed).solve(self.rhs)
        self.reference = np.dot(self.line_weights, np.abs(self.bare[self.line_dofs])**2)
        if not np.isfinite(self.reference) or self.reference <= np.finfo(float).tiny:
            raise RuntimeError("Référence pondérée sur la ligne focale nulle ou non finie.")

    def assemble(self, dofs, local):
        count = len(dofs)
        rows = np.broadcast_to(dofs[:, :, None], (count, 4, 4)).ravel()
        cols = np.broadcast_to(dofs[:, None, :], (count, 4, 4)).ravel()
        values = np.broadcast_to(local, (count, 4, 4)).ravel()
        return sparse.coo_matrix((values, (rows, cols)), shape=(self.size, self.size)).tocsr()

    def solve(self, heights):
        c = self.c
        h = np.asarray(heights, dtype=np.float64).ravel()
        if len(h) != len(self.design_dofs) or not np.isfinite(h).all() or np.any(h < 0) or np.any(h > c.MAX_HEIGHT):
            raise ValueError("Hauteurs continues incompatibles avec la zone de conception.")
        # Two identical sets of columns add mass on opposite faces.
        ratio = 2*c.STUB_DENSITY*c.STUB_DZ/(c.DENSITY*c.THICKNESS)
        delta = self.k0**2*(np.sqrt(1+ratio*h)-1)
        derivative = self.k0**2*ratio/(2*np.sqrt(1+ratio*h))
        values = (-delta[:, None, None]*self.me).ravel()
        patch = sparse.coo_matrix((values, (self.rows, self.cols)),
                                  shape=(self.size, self.size)).tocsc()
        lu = splu(self.fixed+patch)
        field = lu.solve(self.rhs)
        if not np.isfinite(field).all():
            raise RuntimeError("Champ FEM réduit non fini.")
        return field, lu, derivative


class DifferentiableSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heights, plate):
        data = heights.detach().cpu().numpy()
        field, lu, derivative = plate.solve(data)
        ctx.plate, ctx.lu, ctx.derivative, ctx.field = plate, lu, derivative, field
        ctx.shape, ctx.device, ctx.dtype = heights.shape, heights.device, heights.dtype
        return torch.as_tensor(field, dtype=torch.complex128, device=heights.device)

    @staticmethod
    def backward(ctx, grad_field):
        plate = ctx.plate
        adjoint = ctx.lu.solve(grad_field.detach().cpu().numpy(), trans="H")
        local_field = ctx.field[plate.design_dofs]
        local_adj = adjoint[plate.design_dofs]
        grad = ctx.derivative*np.real(np.sum(local_adj.conj()*(local_field @ plate.me.T), axis=1))
        gradient = torch.as_tensor(grad.reshape(ctx.shape), dtype=ctx.dtype, device=ctx.device)
        return gradient, None
