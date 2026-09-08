"""
Extract critical points (zeros) of a piecewise-linear 2D vector field
stored on a regular vtkImageData grid, for a set of timesteps.

For each quad cell the field is split into two triangles; on each triangle
the field is affine, so a zero (if it exists) is found analytically via
barycentric coordinates.  The type of the critical point (source / sink /
saddle) is derived from the sign/trace of the (constant, per-triangle)
Jacobian of the affine map.
"""
import numpy as np
import vtk
from vtk.util import numpy_support as vns


def load_uv(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(path)
    r.Update()
    d = r.GetOutput()
    nx, ny, nz = d.GetDimensions()
    ox, oy, oz = d.GetOrigin()
    sx, sy, sz = d.GetSpacing()
    pd = d.GetPointData()
    u = vns.vtk_to_numpy(pd.GetArray('u')).reshape(ny, nx)
    v = vns.vtk_to_numpy(pd.GetArray('v')).reshape(ny, nx)
    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    return u, v, xs, ys


def _solve_tri(u0, v0, u1, v1, u2, v2, eps=1e-9):
    det = (u0 - u2) * (v1 - v2) - (u1 - u2) * (v0 - v2)
    ok = np.abs(det) > eps
    detc = np.where(ok, det, 1.0)
    l0 = (-u2 * (v1 - v2) + v2 * (u1 - u2)) / detc
    l1 = (-v2 * (u0 - u2) + u2 * (v0 - v2)) / detc
    l2 = 1.0 - l0 - l1
    tol = 1e-6
    inside = ok & (l0 >= -tol) & (l0 <= 1 + tol) & (l1 >= -tol) & (l1 <= 1 + tol) & (l2 >= -tol) & (l2 <= 1 + tol)
    return inside, l0, l1, l2


def _jacobian(u0, v0, u1, v1, u2, v2, e1x, e1y, e2x, e2y):
    det_e = e1x * e2y - e1y * e2x
    du1 = u1 - u0
    du2 = u2 - u0
    dv1 = v1 - v0
    dv2 = v2 - v0
    gux = (du1 * e2y - du2 * e1y) / det_e
    guy = (du2 * e1x - du1 * e2x) / det_e
    gvx = (dv1 * e2y - dv2 * e1y) / det_e
    gvy = (dv2 * e1x - dv1 * e2x) / det_e
    jdet = gux * gvy - guy * gvx
    jtrace = gux + gvy
    return jdet, jtrace


def classify(jdet, jtrace):
    # 0 = saddle, 1 = source (repelling), 2 = sink (attracting)
    if jdet < 0:
        return 0
    return 2 if jtrace < 0 else 1


def extract_critical_points(path):
    u, v, xs, ys = load_uv(path)
    ny, nx = u.shape
    pts = []
    types = []

    # corner values, shape (ny-1, nx-1)
    u00, u10, u11, u01 = u[:-1, :-1], u[:-1, 1:], u[1:, 1:], u[1:, :-1]
    v00, v10, v11, v01 = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    x0 = xs[ii]
    y0 = ys[jj]

    # triangle A: p0=(0,0) p1=(1,0) p2=(1,1)  (local unit-cell coords)
    insideA, l0A, l1A, l2A = _solve_tri(u00, v00, u10, v10, u11, v11)
    jdetA, jtraceA = _jacobian(u00, v00, u10, v10, u11, v11, dx, 0.0, dx, dy)

    # triangle B: p0=(0,0) p1=(1,1) p2=(0,1)
    insideB, l0B, l1B, l2B = _solve_tri(u00, v00, u11, v11, u01, v01)
    jdetB, jtraceB = _jacobian(u00, v00, u11, v11, u01, v01, dx, dy, 0.0, dy)

    for mask, l0, l1, l2, px0, px1, py0, py1, px2, py2, jdet, jtrace in (
        (insideA, l0A, l1A, l2A, x0, x0 + dx, y0, y0, x0 + dx, y0 + dy, jdetA, jtraceA),
        (insideB, l0B, l1B, l2B, x0, x0 + dx, y0, y0 + dy, x0, y0 + dy, jdetB, jtraceB),
    ):
        idxs = np.argwhere(mask)
        for (j, i) in idxs:
            L0, L1, L2 = l0[j, i], l1[j, i], l2[j, i]
            X = L0 * px0[j, i] + L1 * px1[j, i] + L2 * px2[j, i]
            Y = L0 * py0[j, i] + L1 * py1[j, i] + L2 * py2[j, i]
            pts.append((X, Y, 0.0))
            types.append(classify(jdet[j, i], jtrace[j, i]))

    return np.array(pts), np.array(types)


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    pts, types = extract_critical_points(p)
    print(p, '->', len(pts), 'critical points')
    for t in (0, 1, 2):
        print('  type', t, (types == t).sum())
