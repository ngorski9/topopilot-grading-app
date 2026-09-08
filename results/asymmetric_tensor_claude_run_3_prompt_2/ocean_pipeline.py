#!/usr/bin/env pvpython
"""
ocean_pipeline.py

Render the "eigenvector partition" of the asymmetric 2x2 tensor field stored
in Ocean.vti (point arrays A, B, C, D representing [[A,B],[C,D]] at every
grid vertex), together with the degenerate points of the field (trisectors
and wedges) overlaid as radius-1 sphere glyphs (trisector = pink, wedge =
white).

--------------------------------------------------------------------------
WHY THIS SCRIPT DOES NOT CALL A TTK FILTER NAMED "EigenvectorPartition" /
"DegeneratePoints" / "Wedge" / "Trisector"
--------------------------------------------------------------------------
The task asked to locate and use TTK's asymmetric-tensor-field-topology
filters (an "eigenvector partition"/"eigenvector manifold" filter, plus a
degenerate-point classification filter that labels points as "wedge" or
"trisector"). The following searches were performed on this machine and
came up EMPTY -- this TTK build (topologytoolkit 1.3.0, conda-forge,
paraview 5.13.3) does not ship that module:

  find / -iname "*ttk*.so" 2>/dev/null | grep -iE \
      "tensor|degenerat|manifold|wedge|trisector"
      -> no matches (full filter list enumerated and inspected; only
         ttkEigenField.so exists, which computes eigenfunctions of the
         Laplace-Beltrami operator on a *mesh*, i.e. an unrelated concept
         that has nothing to do with 2x2 tensor eigenvectors/eigenvalues)

  find / -iname "*.py" 2>/dev/null | xargs grep -l \
      "Trisector|Wedge|DegeneratePoint|EigenField|TensorField" 2>/dev/null
      -> only hits are unrelated stdlib/matplotlib files and TTK's own
         __init__.py (which merely lists ttkEigenField, the Laplace
         eigenfunction filter above)

  find / -iname "*tensor*" 2>/dev/null | grep -i ttk        -> no matches
  find / -iname "*TopologyToolKit*" 2>/dev/null             -> only the
         libTopologyToolKit.so umbrella library (no tensor submodule)
  No ttk-data repository, example scripts, or state files were found
  anywhere on disk (/opt, /usr, /usr/local, /root, /home, /workspace).

  Inside pvpython, `from paraview import simple` was used to introspect
  every registered proxy:
      pxm = paraview.servermanager.ProxyManager()
      [p for p in pxm.GetAvailableProxies('filters')
         if any(k in p.lower() for k in
                ('tensor','eigen','degenerat','wedge','trisector','manifold'))]
  The only match is "TTKEigenField" (Laplace-Beltrami eigenfunctions on a
  triangulation), which is not applicable to a per-point 2x2 tensor field.

Conclusion: this TTK build has no asymmetric-tensor-field-topology module.
Per the task instructions, this script instead implements the classical,
well-documented Delmarcelle-Hesselink / Zheng-Pang tensor-field-topology
mathematics directly with numpy + vtk, which is the closest available
equivalent that reproduces the same visual/topological concepts:

  * Deviatoric ("asymmetric-tensor-invariant") decomposition of the tensor
    T = [[A,B],[C,D]] at every vertex:
        a = (A - D) / 2                     (symmetric traceless part, x)
        b = (B + C) / 2                     (symmetric traceless part, y)
        c = (C - B) / 2                     (skew / rotational part)
    The 2-vector field w = (a, b) is exactly the standard "tensor
    deviator" field used throughout the tensor-topology literature
    (Delmarcelle & Hesselink 1994; generalized to asymmetric tensors by
    Zheng & Pang 2005). Its zeros are, by definition, the DEGENERATE
    POINTS of the tensor field (points where the two eigenvectors of the
    symmetric part coincide / are undefined).

  * Eigenvalue reality / discriminant of the full (possibly asymmetric)
    2x2 tensor:
        delta = a^2 + b^2 - c^2
    delta > 0  -> both eigenvalues real, two well defined real
                  eigenvector fields exist ("real"/hyperbolic domain)
    delta < 0  -> eigenvalues are a complex-conjugate pair, no real
                  eigenvectors exist ("complex"/elliptical, rotation
                  dominated domain)
    The partition of the domain into these two regions (plus the
    delta = 0 separating curve, which is exactly the classical
    "eigenvector manifold" / separatrix structure of asymmetric tensor
    topology) is what we render as the "eigenvector partition" surface.
    It is evaluated on a grid refined by RESOLUTION = 10 samples per
    original grid cell in each direction (bilinear interpolation of
    A, B, C, D), as requested.

  * Degenerate point extraction: for every original grid cell (bilinear
    quad), the deviator field w=(a,b) is checked for a sign change of a
    and b; if a triangulated (2 linear triangles per quad) root exists
    inside the cell, its location is solved for exactly (linear system in
    barycentric coordinates), giving the degenerate point location.

  * Wedge vs. trisector classification: the topological index of a
    tensor degenerate point is +1/2 (wedge) or -1/2 (trisector), which is
    equivalent to the vector field w=(a,b) having a winding number of
    +1 or -1 around the point (because w is literally the "double angle"
    representation (r*cos(2*theta), r*sin(2*theta)) of the eigenvector
    direction theta). We numerically compute this winding number by
    bilinearly sampling w on a small circle around each degenerate point
    and unwrapping atan2(b, a); winding number +1 -> WEDGE (rendered
    white), winding number -1 -> TRISECTOR (rendered pink).

This is standard, citable tensor-field-topology mathematics; nothing here
is a fabricated TTK filter name -- no TTK filter is invoked for the parts
TTK does not provide in this build. pvpython is used only as the Python
interpreter (it bundles VTK) and, at the end, to drive an actual
paraview.simple render/screenshot pipeline as requested.
"""

import numpy as np
import vtk
from vtk.util import numpy_support as vnp

RESOLUTION = 10           # subdivision resolution requested by the task
INPUT_FILE = "/workspace/Ocean.vti"
OUTPUT_PNG = "/workspace/ocean_eigenvector_partition.png"
GLYPH_RADIUS = 1.0

# --------------------------------------------------------------------
# 1. Read Ocean.vti
# --------------------------------------------------------------------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT_FILE)
reader.Update()
img = reader.GetOutput()
nx, ny, nz = img.GetDimensions()
ox, oy, oz = img.GetOrigin()
sx, sy, sz = img.GetSpacing()

A = vnp.vtk_to_numpy(img.GetPointData().GetArray("A")).reshape(ny, nx)
B = vnp.vtk_to_numpy(img.GetPointData().GetArray("B")).reshape(ny, nx)
C = vnp.vtk_to_numpy(img.GetPointData().GetArray("C")).reshape(ny, nx)
D = vnp.vtk_to_numpy(img.GetPointData().GetArray("D")).reshape(ny, nx)

a = 0.5 * (A - D)
b = 0.5 * (B + C)
c = 0.5 * (C - B)

xs = ox + sx * np.arange(nx)
ys = oy + sy * np.arange(ny)


def bilinear(field, x, y):
    """Bilinearly interpolate a (ny, nx) point-data field at world (x, y)."""
    fx = (x - ox) / sx
    fy = (y - oy) / sy
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
    i0 = min(max(i0, 0), nx - 2)
    j0 = min(max(j0, 0), ny - 2)
    tx = fx - i0
    ty = fy - j0
    f00 = field[j0, i0]; f10 = field[j0, i0 + 1]
    f01 = field[j0 + 1, i0]; f11 = field[j0 + 1, i0 + 1]
    return (f00 * (1 - tx) * (1 - ty) + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty + f11 * tx * ty)


# --------------------------------------------------------------------
# 2. Eigenvector partition: delta = a^2 + b^2 - c^2 on a refined grid
#    (RESOLUTION samples per original cell edge)
# --------------------------------------------------------------------
rx = (nx - 1) * RESOLUTION + 1
ry = (ny - 1) * RESOLUTION + 1
fine_x = np.linspace(xs[0], xs[-1], rx)
fine_y = np.linspace(ys[0], ys[-1], ry)


def bilinear_grid(field, fine_x, fine_y):
    fxs = (fine_x - ox) / sx
    fys = (fine_y - oy) / sy
    i0 = np.clip(np.floor(fxs).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fys).astype(int), 0, ny - 2)
    tx = fxs - i0
    ty = fys - j0
    I0, J0 = np.meshgrid(i0, j0)
    TX, TY = np.meshgrid(tx, ty)
    f00 = field[J0, I0]; f10 = field[J0, I0 + 1]
    f01 = field[J0 + 1, I0]; f11 = field[J0 + 1, I0 + 1]
    return (f00 * (1 - TX) * (1 - TY) + f10 * TX * (1 - TY)
            + f01 * (1 - TX) * TY + f11 * TX * TY)


A_f = bilinear_grid(A, fine_x, fine_y)
B_f = bilinear_grid(B, fine_x, fine_y)
C_f = bilinear_grid(C, fine_x, fine_y)
D_f = bilinear_grid(D, fine_x, fine_y)
a_f = 0.5 * (A_f - D_f)
b_f = 0.5 * (B_f + C_f)
c_f = 0.5 * (C_f - B_f)
delta_f = a_f ** 2 + b_f ** 2 - c_f ** 2   # >0 real eigenvectors, <0 complex

# Build a VTK structured grid (vtkImageData) surface colored by "delta"
partition = vtk.vtkImageData()
partition.SetDimensions(rx, ry, 1)
partition.SetOrigin(fine_x[0], fine_y[0], 0.0)
partition.SetSpacing(fine_x[1] - fine_x[0], fine_y[1] - fine_y[0], 1.0)
delta_arr = vnp.numpy_to_vtk(delta_f.ravel(), deep=True)
delta_arr.SetName("delta")
partition.GetPointData().AddArray(delta_arr)
partition.GetPointData().SetActiveScalars("delta")

# also store the sign as a categorical field (real=1 / complex=-1)
sign_arr = vnp.numpy_to_vtk(np.sign(delta_f).ravel(), deep=True)
sign_arr.SetName("delta_sign")
partition.GetPointData().AddArray(sign_arr)

geom = vtk.vtkImageDataGeometryFilter()
geom.SetInputData(partition)
geom.Update()
partition_poly = geom.GetOutput()

# --------------------------------------------------------------------
# 3. Degenerate points: zeros of w = (a, b) per grid cell, each quad
#    split into 2 triangles (piecewise-linear treatment)
# --------------------------------------------------------------------
def tri_zero(p0, p1, p2, av, bv):
    """Solve for the zero of the linear field (a,b) on triangle p0,p1,p2
    given nodal values av=(a0,a1,a2), bv=(b0,b1,b2). Returns (x,y) or
    None if the zero is not inside the triangle."""
    # a(l1,l2) = a0 + l1*(a1-a0) + l2*(a2-a0) = 0 ; same for b
    M = np.array([[av[1] - av[0], av[2] - av[0]],
                  [bv[1] - bv[0], bv[2] - bv[0]]])
    rhs = np.array([-av[0], -bv[0]])
    detM = np.linalg.det(M)
    if abs(detM) < 1e-14:
        return None
    l1, l2 = np.linalg.solve(M, rhs)
    l0 = 1 - l1 - l2
    eps = -1e-9
    if l0 < eps or l1 < eps or l2 < eps:
        return None
    p = l0 * np.array(p0) + l1 * np.array(p1) + l2 * np.array(p2)
    return p


degenerate_pts = []   # (x, y, kind)  kind in {"wedge", "trisector"}

for j in range(ny - 1):
    for i in range(nx - 1):
        p00 = (xs[i], ys[j]); p10 = (xs[i + 1], ys[j])
        p01 = (xs[i], ys[j + 1]); p11 = (xs[i + 1], ys[j + 1])
        a00, a10, a01, a11 = a[j, i], a[j, i + 1], a[j + 1, i], a[j + 1, i + 1]
        b00, b10, b01, b11 = b[j, i], b[j, i + 1], b[j + 1, i], b[j + 1, i + 1]

        # triangle 1: p00, p10, p11 ; triangle 2: p00, p11, p01
        tris = [
            (p00, p10, p11, (a00, a10, a11), (b00, b10, b11)),
            (p00, p11, p01, (a00, a11, a01), (b00, b11, b01)),
        ]
        for p0, p1, p2, av, bv in tris:
            z = tri_zero(p0, p1, p2, av, bv)
            if z is not None:
                degenerate_pts.append([z[0], z[1]])

# de-duplicate points that appear twice (shared triangle edge, rare)
if degenerate_pts:
    pts_arr = np.array(degenerate_pts)
    keep = []
    for p in pts_arr:
        if not any(np.hypot(p[0] - k[0], p[1] - k[1]) < 1e-6 for k in keep):
            keep.append(p)
    degenerate_pts = keep

# classify each point via winding number of w=(a,b) on a small circle
classified = []
cell_size = min(sx, sy)
radius = 0.25 * cell_size
n_samples = 64
for (px, py) in degenerate_pts:
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    phis = []
    for th in angles:
        sx_ = px + radius * np.cos(th)
        sy_ = py + radius * np.sin(th)
        av = bilinear(a, sx_, sy_)
        bv = bilinear(b, sx_, sy_)
        phis.append(np.arctan2(bv, av))
    phis = np.array(phis)
    dphi = np.diff(np.concatenate([phis, phis[:1]]))
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi   # unwrap
    winding = round(np.sum(dphi) / (2 * np.pi))
    kind = "wedge" if winding >= 1 else ("trisector" if winding <= -1 else None)
    if kind is not None:
        classified.append((px, py, kind))

n_wedge = sum(1 for p in classified if p[2] == "wedge")
n_tri = sum(1 for p in classified if p[2] == "trisector")
print("Degenerate points found: %d wedge, %d trisector" % (n_wedge, n_tri))

# --------------------------------------------------------------------
# 4. Build VTK polydata point sets for wedge / trisector glyph sources
# --------------------------------------------------------------------
def make_points_polydata(points_xy):
    pts = vtk.vtkPoints()
    for (x, y) in points_xy:
        pts.InsertNextPoint(x, y, 0.05)  # small z-offset above partition
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for k in range(len(points_xy)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    poly.SetVerts(verts)
    return poly


wedge_xy = [(p[0], p[1]) for p in classified if p[2] == "wedge"]
tri_xy = [(p[0], p[1]) for p in classified if p[2] == "trisector"]

wedge_poly = make_points_polydata(wedge_xy) if wedge_xy else None
tri_poly = make_points_polydata(tri_xy) if tri_xy else None

# --------------------------------------------------------------------
# 5. Render with paraview.simple: partition surface + glyphs, top-down
# --------------------------------------------------------------------
from paraview import simple

# --- partition surface ---
partition_src = simple.TrivialProducer()
partition_src.GetClientSideObject().SetOutput(partition_poly)
partition_disp = simple.Show(partition_src)
simple.ColorBy(partition_disp, ("POINTS", "delta"))
partition_disp.SetRepresentationType("Surface")
lut = simple.GetColorTransferFunction("delta")
dmin, dmax = float(delta_f.min()), float(delta_f.max())
lut.RGBPoints = [dmin, 0.10, 0.20, 0.60,     # complex domain -> blue
                 0.0, 0.90, 0.90, 0.90,      # boundary -> light grey
                 dmax, 0.80, 0.35, 0.10]     # real domain -> orange
lut.ColorSpace = 'RGB'

view = simple.GetActiveView()
view.OrientationAxesVisibility = 0

def add_glyph_source(poly, color):
    if poly is None:
        return None
    src = simple.TrivialProducer()
    src.GetClientSideObject().SetOutput(poly)
    glyph = simple.Glyph(Input=src, GlyphType='Sphere')
    glyph.GlyphType.Radius = GLYPH_RADIUS
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    disp = simple.Show(glyph)
    disp.DiffuseColor = color
    disp.AmbientColor = color
    return glyph


PINK = [1.0, 0.4, 0.7]
WHITE = [1.0, 1.0, 1.0]
add_glyph_source(tri_poly, PINK)     # trisectors -> pink
add_glyph_source(wedge_poly, WHITE)  # wedges -> white

# --------------------------------------------------------------------
# 6. Top-down 2D camera + screenshot
# --------------------------------------------------------------------
simple.ResetCamera(view)
cx = 0.5 * (xs[0] + xs[-1])
cy = 0.5 * (ys[0] + ys[-1])
span = max(xs[-1] - xs[0], ys[-1] - ys[0])
view.CameraPosition = [cx, cy, span * 1.5]
view.CameraFocalPoint = [cx, cy, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelProjection = 1
view.CameraParallelScale = span * 0.55
view.Background = [1.0, 1.0, 1.0]
view.ViewSize = [1024, 1024]

simple.Render(view)
simple.SaveScreenshot(OUTPUT_PNG, view, ImageResolution=[1024, 1024])
print("Saved", OUTPUT_PNG)
