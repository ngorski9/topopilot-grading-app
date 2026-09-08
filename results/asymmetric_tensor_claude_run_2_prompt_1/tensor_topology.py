import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk

from paraview.simple import *

# ---------------------------------------------------------------
# 1. Load the asymmetric tensor field Ocean.vti (components A,B,C,D
#    represent the 2x2 matrix [[A, B], [C, D]] at each grid vertex)
# ---------------------------------------------------------------
reader = XMLImageDataReader(FileName=['/workspace/Ocean.vti'])
reader.UpdatePipeline()
img = reader.GetClientSideObject().GetOutputDataObject(0)

dims = img.GetDimensions()
spacing = img.GetSpacing()
origin = img.GetOrigin()
nx, ny = dims[0], dims[1]

pd = img.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
C = vtk_to_numpy(pd.GetArray('C')).reshape(ny, nx)
D = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

x0 = np.arange(nx) * spacing[0] + origin[0]
y0 = np.arange(ny) * spacing[1] + origin[1]

# ---------------------------------------------------------------
# 2. Resample to the requested resolution: 10 pixels per unit square
# ---------------------------------------------------------------
PPU = 10.0
Wx = (nx - 1) * spacing[0]
Wy = (ny - 1) * spacing[1]
fnx = int(round(Wx * PPU)) + 1
fny = int(round(Wy * PPU)) + 1

from scipy.interpolate import RegularGridInterpolator

def upsample(F):
    interp = RegularGridInterpolator((y0, x0), F, method='linear')
    return interp

xf = np.linspace(x0[0], x0[-1], fnx)
yf = np.linspace(y0[0], y0[-1], fny)
XX, YY = np.meshgrid(xf, yf)  # shape (fny, fnx)
pts = np.stack([YY.ravel(), XX.ravel()], axis=1)

Af = upsample(A)(pts).reshape(fny, fnx)
Bf = upsample(B)(pts).reshape(fny, fnx)
Cf = upsample(C)(pts).reshape(fny, fnx)
Df = upsample(D)(pts).reshape(fny, fnx)

E = Af - Df          # deviator component 1
Fc = Bf + Cf         # deviator component 2
delta = 0.25 * E ** 2 + Bf * Cf   # ((A-D)/2)^2 + B*C

# ---------------------------------------------------------------
# 3. Eigenvector partition: real (delta>=0) vs complex (delta<0) domain
# ---------------------------------------------------------------
partition = (delta >= 0).astype(np.float64)  # 1 = real eigenvector domain, 0 = complex domain

# ---------------------------------------------------------------
# 4. Degenerate points: common zeros of E and Fc (cell-wise bilinear solve)
# ---------------------------------------------------------------
def cell_sign_changes(F):
    c00 = F[:-1, :-1]
    c10 = F[:-1, 1:]
    c01 = F[1:, :-1]
    c11 = F[1:, 1:]
    pos = (c00 > 0) | (c10 > 0) | (c01 > 0) | (c11 > 0)
    neg = (c00 < 0) | (c10 < 0) | (c01 < 0) | (c11 < 0)
    return pos & neg

E_change = cell_sign_changes(E)
F_change = cell_sign_changes(Fc)
candidates = np.argwhere(E_change & F_change)  # (row, col) of lower-left corner

dx = xf[1] - xf[0]
dy = yf[1] - yf[0]

def bilinear(F, i, j, u, v):
    c00 = F[i, j]; c10 = F[i, j+1]; c01 = F[i+1, j]; c11 = F[i+1, j+1]
    return (c00 * (1 - u) * (1 - v) + c10 * u * (1 - v)
            + c01 * (1 - u) * v + c11 * u * v)

def bilinear_grad(F, i, j, u, v):
    c00 = F[i, j]; c10 = F[i, j+1]; c01 = F[i+1, j]; c11 = F[i+1, j+1]
    dFdu = (c10 - c00) * (1 - v) + (c11 - c01) * v
    dFdv = (c01 - c00) * (1 - u) + (c11 - c10) * u
    return dFdu, dFdv

degenerate_pts = []
for (i, j) in candidates:
    u, v = 0.5, 0.5
    ok = True
    for _ in range(20):
        e_val = bilinear(E, i, j, u, v)
        f_val = bilinear(Fc, i, j, u, v)
        dEdu, dEdv = bilinear_grad(E, i, j, u, v)
        dFdu, dFdv = bilinear_grad(Fc, i, j, u, v)
        Jmat = np.array([[dEdu, dEdv], [dFdu, dFdv]])
        det = Jmat[0, 0] * Jmat[1, 1] - Jmat[0, 1] * Jmat[1, 0]
        if abs(det) < 1e-14:
            ok = False
            break
        delta_uv = np.linalg.solve(Jmat, -np.array([e_val, f_val]))
        u += delta_uv[0]
        v += delta_uv[1]
        if abs(delta_uv[0]) < 1e-10 and abs(delta_uv[1]) < 1e-10:
            break
    if not ok or not (-1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6):
        continue
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)
    x = xf[j] + u * dx
    y = yf[i] + v * dy

    # classify via sign of Jacobian determinant of (E, Fc) at this location
    dEdu, dEdv = bilinear_grad(E, i, j, u, v)
    dFdu, dFdv = bilinear_grad(Fc, i, j, u, v)
    # convert du,dv derivatives (unit cell) to physical dx,dy derivatives
    dEdx, dEdy = dEdu / dx, dEdv / dy
    dFdx, dFdy = dFdu / dx, dFdv / dy
    det = dEdx * dFdy - dEdy * dFdx
    kind = 'wedge' if det > 0 else 'trisector'
    degenerate_pts.append((x, y, kind))

# merge points that are closer than one fine-grid cell (duplicate detections
# from neighboring flagged cells)
merged = []
used = [False] * len(degenerate_pts)
for idx, (x, y, kind) in enumerate(degenerate_pts):
    if used[idx]:
        continue
    group = [(x, y, kind)]
    used[idx] = True
    for jdx in range(idx + 1, len(degenerate_pts)):
        if used[jdx]:
            continue
        x2, y2, k2 = degenerate_pts[jdx]
        if abs(x2 - x) < 2 * dx and abs(y2 - y) < 2 * dy:
            group.append((x2, y2, k2))
            used[jdx] = True
    gx = np.mean([g[0] for g in group])
    gy = np.mean([g[1] for g in group])
    gk = group[0][2]
    merged.append((gx, gy, gk))

wedges = [(x, y) for x, y, k in merged if k == 'wedge']
trisectors = [(x, y) for x, y, k in merged if k == 'trisector']
print('Found %d wedges, %d trisectors' % (len(wedges), len(trisectors)))

# ---------------------------------------------------------------
# 5. Build a vtkImageData holding the eigenvector partition scalar field
# ---------------------------------------------------------------
part_img = vtk.vtkImageData()
part_img.SetDimensions(fnx, fny, 1)
part_img.SetSpacing(dx, dy, 1.0)
part_img.SetOrigin(xf[0], yf[0], 0.0)
arr = numpy_to_vtk(partition.ravel(), deep=True)
arr.SetName('EigenvectorPartition')
part_img.GetPointData().SetScalars(arr)

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName('/workspace/eigenvector_partition.vti')
writer.SetInputData(part_img)
writer.Write()

def write_points(pts_list, fname):
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for (x, y) in pts_list:
        pid = points.InsertNextPoint(x, y, 0.0)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(pid)
    poly.SetPoints(points)
    poly.SetVerts(verts)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(fname)
    w.SetInputData(poly)
    w.Write()

write_points(wedges, '/workspace/wedges.vtp')
write_points(trisectors, '/workspace/trisectors.vtp')

print('done')
