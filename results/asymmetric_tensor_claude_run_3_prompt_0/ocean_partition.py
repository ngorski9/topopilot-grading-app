#!/usr/bin/env pvpython
"""
Eigenvector partition + degenerate point (wedge/trisector) extraction
for a 2x2 asymmetric tensor field defined on Ocean.vti (point arrays A,B,C,D).

Run with:  pvpython --force-offscreen-rendering ocean_partition.py
"""
import numpy as np
import vtk
from vtk.util import numpy_support as nps

INFILE = "/workspace/Ocean.vti"
RES = 10          # pixels per grid-cell for the partition raster
RADIUS = 1.0       # loop radius (in grid-cell units) for Poincare index

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INFILE)
reader.Update()
img = reader.GetOutput()
nx, ny, nz = img.GetDimensions()
origin = img.GetOrigin()
spacing = img.GetSpacing()
print("dims:", nx, ny, nz, "origin:", origin, "spacing:", spacing)

pd = img.GetPointData()
A = nps.vtk_to_numpy(pd.GetArray("A")).reshape(ny, nx).astype(np.float64)
B = nps.vtk_to_numpy(pd.GetArray("B")).reshape(ny, nx).astype(np.float64)
C = nps.vtk_to_numpy(pd.GetArray("C")).reshape(ny, nx).astype(np.float64)
D = nps.vtk_to_numpy(pd.GetArray("D")).reshape(ny, nx).astype(np.float64)
# arrays indexed [row=j (y), col=i (x)]

E = 0.5 * (A - D)          # symmetric deviator term
F = 0.5 * (B + C)          # symmetric off-diagonal term


def bilinear_eval(field, i0, j0, s, t):
    """Bilinear interpolation of `field` (ny,nx) at fractional cell coords.
    i0,j0 = integer base cell indices (may be out of range -> clamped),
    s,t in general reals (extrapolation allowed by clamping corner index
    but using the *linear* extension of the same cell's bilinear form)."""
    i0c = min(max(i0, 0), nx - 2)
    j0c = min(max(j0, 0), ny - 2)
    f00 = field[j0c, i0c]
    f10 = field[j0c, i0c + 1]
    f01 = field[j0c + 1, i0c]
    f11 = field[j0c + 1, i0c + 1]
    return (f00 * (1 - s) * (1 - t) + f10 * s * (1 - t) +
            f01 * (1 - s) * t + f11 * s * t)


def EF_at(x, y):
    """Evaluate (E,F) at continuous grid coordinates (x,y) in [0,nx-1]x[0,ny-1],
    using the cell that contains (x,y); if outside domain, clamp to nearest
    cell and allow s,t to go slightly out of [0,1] for a smooth extension."""
    i0 = int(np.floor(x))
    j0 = int(np.floor(y))
    i0 = min(max(i0, 0), nx - 2)
    j0 = min(max(j0, 0), ny - 2)
    s = x - i0
    t = y - j0
    e = bilinear_eval(E, i0, j0, s, t)
    f = bilinear_eval(F, i0, j0, s, t)
    return e, f


# ----------------------------------------------------------------------
# 2. Degenerate point detection: per-cell bilinear zero of (E,F)
# ----------------------------------------------------------------------
# For a quad with bilinear scalar field g(s,t) = g00(1-s)(1-t)+g10 s(1-t)+g01(1-s)t+g11 st,
# solve simultaneously E(s,t)=0 and F(s,t)=0 for (s,t) in [0,1]^2.
# Standard approach: write both as bilinear forms
#   g(s,t) = a + b*s + c*t + d*s*t
# with a=g00, b=g10-g00, c=g01-g00, d=g11-g10-g01+g00
# Two such equations (for E and F). Eliminate one variable (resultant in s)
# giving a quadratic in s; solve, back-substitute for t, check ranges.

def bilinear_coeffs(g00, g10, g01, g11):
    a = g00
    b = g10 - g00
    c = g01 - g00
    d = g11 - g10 - g01 + g00
    return a, b, c, d


def solve_cell_zero(i, j):
    """Return list of (s,t) roots in [0,1]x[0,1] of E=F=0 within cell (i,j)."""
    e00, e10, e01, e11 = E[j, i], E[j, i + 1], E[j + 1, i], E[j + 1, i + 1]
    f00, f10, f01, f11 = F[j, i], F[j, i + 1], F[j + 1, i], F[j + 1, i + 1]
    ae, be, ce, de = bilinear_coeffs(e00, e10, e01, e11)
    af, bf, cf, df = bilinear_coeffs(f00, f10, f01, f11)

    # E(s,t)=0 -> (ae+be*s) + (ce+de*s)*t = 0  => t = -(ae+be*s)/(ce+de*s)  (if denom != 0)
    # Substitute into F(s,t)=0:
    #   af + bf*s + (cf+df*s)*t = 0
    # => (af+bf*s)*(ce+de*s) - (ae+be*s)*(cf+df*s) = 0   [multiplying through by (ce+de*s)]
    # This is a quadratic in s.
    # coefficients: expand
    # P(s) = (af+bf*s)(ce+de*s) - (ae+be*s)(cf+df*s)
    p2 = bf * de - be * df
    p1 = af * de + bf * ce - (ae * df + be * cf)
    p0 = af * ce - ae * cf

    roots_s = []
    if abs(p2) < 1e-14:
        if abs(p1) > 1e-14:
            roots_s = [-p0 / p1]
    else:
        disc = p1 * p1 - 4 * p2 * p0
        if disc >= 0:
            sq = np.sqrt(disc)
            roots_s = [(-p1 + sq) / (2 * p2), (-p1 - sq) / (2 * p2)]

    results = []
    for s in roots_s:
        if -1e-9 <= s <= 1 + 1e-9:
            denom = ce + de * s
            if abs(denom) < 1e-12:
                continue
            t = -(ae + be * s) / denom
            if -1e-9 <= t <= 1 + 1e-9:
                results.append((min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)))
    return results


def poincare_index(x, y, radius=RADIUS, nsamp=64):
    angles = []
    for k in range(nsamp):
        th = 2 * np.pi * k / nsamp
        xx = x + radius * np.cos(th)
        yy = y + radius * np.sin(th)
        e, f = EF_at(xx, yy)
        angles.append(np.arctan2(f, e))
    angles = np.array(angles)
    d = np.diff(np.concatenate([angles, angles[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi,pi]
    total = np.sum(d)
    # NOTE: (E,F) is the "double-angle" representation of the eigenvector
    # direction field (E,F) ~ (cos(2*theta), sin(2*theta)) * magnitude, so its
    # winding number is TWICE the eigenvector-field topological index.
    # Divide by 2 to get the standard tensor degenerate-point index
    # (+1/2 = wedge, -1/2 = trisector).
    index = total / (2 * np.pi) / 2.0
    return index


deg_points = []  # (x,y in index-coords, index_value, kind)
for j in range(ny - 1):
    for i in range(nx - 1):
        roots = solve_cell_zero(i, j)
        for (s, t) in roots:
            x = i + s
            y = j + t
            idx = poincare_index(x, y)
            rounded = round(idx * 2) / 2.0
            if abs(rounded - 0.5) < 0.26:
                kind = 1  # wedge
            elif abs(rounded + 0.5) < 0.26:
                kind = 0  # trisector
            else:
                continue  # skip index-0 or ambiguous (numerical noise)
            deg_points.append((x, y, rounded, kind))

# de-duplicate points that are extremely close (shared cell edges/corners)
dedup = []
for p in deg_points:
    dup = False
    for q in dedup:
        if abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6:
            dup = True
            break
    if not dup:
        dedup.append(p)
deg_points = dedup

n_wedge = sum(1 for p in deg_points if p[3] == 1)
n_tri = sum(1 for p in deg_points if p[3] == 0)
print("Degenerate points found:", len(deg_points), " wedge:", n_wedge, " trisector:", n_tri)

# ----------------------------------------------------------------------
# 3. Build degenerate-point polydata (in physical/world coords)
# ----------------------------------------------------------------------
pts = vtk.vtkPoints()
type_arr = vtk.vtkIntArray()
type_arr.SetName("Type")
verts = vtk.vtkCellArray()
for (x, y, idxval, kind) in deg_points:
    wx = origin[0] + x * spacing[0]
    wy = origin[1] + y * spacing[1]
    pid = pts.InsertNextPoint(wx, wy, 0.01)
    verts.InsertNextCell(1)
    verts.InsertCellPoint(pid)
    type_arr.InsertNextValue(kind)

deg_poly = vtk.vtkPolyData()
deg_poly.SetPoints(pts)
deg_poly.SetVerts(verts)
deg_poly.GetPointData().AddArray(type_arr)
deg_poly.GetPointData().SetActiveScalars("Type")

# ----------------------------------------------------------------------
# 4. Eigenvector-partition raster image (upsampled RES x per cell)
# ----------------------------------------------------------------------
pnx = (nx - 1) * RES + 1
pny = (ny - 1) * RES + 1

part_img = vtk.vtkImageData()
part_img.SetDimensions(pnx, pny, 1)
part_img.SetOrigin(origin[0], origin[1], 0.0)
part_img.SetSpacing(spacing[0] / RES, spacing[1] / RES, spacing[2])

angle = np.zeros((pny, pnx), dtype=np.float64)

xs = np.linspace(0, nx - 1, pnx)
ys = np.linspace(0, ny - 1, pny)

for jp, y in enumerate(ys):
    j0 = min(int(np.floor(y)), ny - 2)
    t = y - j0
    for ip, x in enumerate(xs):
        i0 = min(int(np.floor(x)), nx - 2)
        s = x - i0
        a = bilinear_eval(A, i0, j0, s, t)
        b = bilinear_eval(B, i0, j0, s, t)
        c = bilinear_eval(C, i0, j0, s, t)
        d = bilinear_eval(D, i0, j0, s, t)
        sxy = 0.5 * (b + c)
        # symmetric part eigen-decomposition (2x2)
        tr = a + d
        e = 0.5 * (a - d)
        # eigenvalues: tr/2 +- sqrt(e^2+sxy^2)
        # eigenvector of larger eigenvalue: angle = 0.5*atan2(2*sxy, a-d)
        th = 0.5 * np.arctan2(2 * sxy, a - d)
        angle[jp, ip] = th % np.pi

angle_flat = angle.reshape(-1)
angle_vtk = nps.numpy_to_vtk(angle_flat, deep=True)
angle_vtk.SetName("EigenvectorAngle")
part_img.GetPointData().AddArray(angle_vtk)
part_img.GetPointData().SetActiveScalars("EigenvectorAngle")

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("/workspace/_partition_debug.vti")
writer.SetInputData(part_img)
writer.Write()

poly_writer = vtk.vtkXMLPolyDataWriter()
poly_writer.SetFileName("/workspace/_degpoints_debug.vtp")
poly_writer.SetInputData(deg_poly)
poly_writer.Write()

# ----------------------------------------------------------------------
# 5. Render with paraview.simple
# ----------------------------------------------------------------------
from paraview.simple import (
    XMLImageDataReader, XMLPolyDataReader, GetActiveViewOrCreate, Show,
    ColorBy, GetColorTransferFunction, Glyph, Sphere,
    Render, SaveScreenshot, ResetCamera, servermanager, RenameSource,
    Hide, GetLayout
)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1000, 1000]
view.OrientationAxesVisibility = 0
view.Background = [0.15, 0.15, 0.15]

part_src = XMLImageDataReader(FileName=["/workspace/_partition_debug.vti"])
RenameSource("PartitionImage", part_src)
part_disp = Show(part_src, view)
ColorBy(part_disp, ('POINTS', 'EigenvectorAngle'))
angle_lut = GetColorTransferFunction('EigenvectorAngle')
angle_lut.ApplyPreset('Rainbow Uniform', True)
angle_lut.RescaleTransferFunction(0.0, np.pi)
part_disp.SetScalarBarVisibility(view, True)

deg_src = XMLPolyDataReader(FileName=["/workspace/_degpoints_debug.vtp"])
RenameSource("DegeneratePoints", deg_src)

sphere = Sphere()
sphere.Radius = spacing[0] * 0.6
sphere.ThetaResolution = 16
sphere.PhiResolution = 16

glyph = Glyph(Input=deg_src, GlyphType=sphere)
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'
glyph_disp = Show(glyph, view)
ColorBy(glyph_disp, ('POINTS', 'Type'))

type_lut = GetColorTransferFunction('Type')
type_lut.InterpretValuesAsCategories = 1
type_lut.AnnotationsInitialized = 1
type_lut.Annotations = ['0', 'Trisector', '1', 'Wedge']
type_lut.IndexedColors = [1.0, 0.75, 0.8,   # pink -> trisector (0)
                           1.0, 1.0, 1.0]   # white -> wedge (1)
glyph_disp.SetScalarBarVisibility(view, False)

ResetCamera(view)
view.CameraParallelProjection = 1
Render(view)

SaveScreenshot("/workspace/ocean_partition.png", view, ImageResolution=[1000, 1000])
servermanager.SaveState("/workspace/ocean_partition.pvsm")

print("DONE")
print("Degenerate points found:", len(deg_points), " wedge:", n_wedge, " trisector:", n_tri)
