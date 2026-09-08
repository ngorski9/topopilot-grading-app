"""
Eigenvector partition + degenerate point extraction/visualization
for the asymmetric piecewise-linear 2x2 tensor field stored in Ocean.vti
(components A, B, C, D -> T = [[A, B], [C, D]]).

Theory (Zheng & Pang, "2D Asymmetric Tensor Field Topology" and
Delmarcelle & Hesselink degenerate-point theory):

  Deviator components of T:
    E = (A - D) / 2
    F = (B + C) / 2
    H = (B - C) / 2        (asymmetric / rotational part)

  Eigenvalues of T are  tr(T)/2 +/- sqrt(E^2 + F^2 - H^2).

  Discriminant  Delta = E^2 + F^2 - H^2
    Delta > 0  -> real, distinct eigenvalues  -> two real eigenvector fields
                  ("real" domain of the eigenvector partition)
    Delta < 0  -> complex-conjugate eigenvalues -> no real eigenvectors
                  ("complex"/rotational domain of the eigenvector partition)
    Delta = 0  -> boundary between the two domains.

  Degenerate points are isolated points where E = 0 and F = 0
  simultaneously (repeated-eigenvalue / isotropic points). Since A,B,C,D
  are piecewise-linear on the grid, E and F are piecewise-linear too, so
  their common zero is found triangle-by-triangle (each quad cell split
  into two triangles) by linear root finding.

  Each degenerate point is classified using the sign of the Jacobian of
  (E, F) with respect to (x, y), which is constant within a linear
  triangle:
     delta = dE/dx * dF/dy - dE/dy * dF/dx
     delta > 0  -> wedge     (Poincare index +1/2)
     delta < 0  -> trisector (Poincare index -1/2)
"""
import numpy as np
from paraview.simple import *
from paraview import servermanager as sm
from vtkmodules.util import numpy_support as ns
import vtk

IN_FILE = "/workspace/Ocean.vti"
BLOCK = 10  # pixels per square of the partition visualization
GLYPH_RADIUS = 1.0

# ---------------------------------------------------------------------------
# 1. Load the tensor field
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=[IN_FILE])
reader.UpdatePipeline()
data = sm.Fetch(reader)

dims = data.GetDimensions()
nx, ny = dims[0], dims[1]
spacing = data.GetSpacing()
origin = data.GetOrigin()

pd = data.GetPointData()
A = ns.vtk_to_numpy(pd.GetArray("A")).reshape(ny, nx)
B = ns.vtk_to_numpy(pd.GetArray("B")).reshape(ny, nx)
C = ns.vtk_to_numpy(pd.GetArray("C")).reshape(ny, nx)
D = ns.vtk_to_numpy(pd.GetArray("D")).reshape(ny, nx)

E = 0.5 * (A - D)
F = 0.5 * (B + C)
H = 0.5 * (B - C)
Delta = E * E + F * F - H * H

# ---------------------------------------------------------------------------
# 2. Eigenvector partition raster, coarsened to BLOCK x BLOCK pixel squares
#    class 0 -> "complex" domain (Delta <= 0, no real eigenvectors)
#    class 1 -> "real" domain   (Delta  > 0, two real eigenvector fields)
# ---------------------------------------------------------------------------
ncellsx = (nx - 1)
ncellsy = (ny - 1)
# per-pixel-cell classification, using the average Delta of the 4 corners
Delta_cell = 0.25 * (Delta[:-1, :-1] + Delta[:-1, 1:] + Delta[1:, :-1] + Delta[1:, 1:])
class_cell = (Delta_cell > 0).astype(np.uint8)

nbx = ncellsx // BLOCK
nby = ncellsy // BLOCK
block_class = np.zeros((nby, nbx), dtype=np.uint8)
for j in range(nby):
    for i in range(nbx):
        blk = class_cell[j * BLOCK:(j + 1) * BLOCK, i * BLOCK:(i + 1) * BLOCK]
        block_class[j, i] = 1 if blk.mean() >= 0.5 else 0

partition_img = vtk.vtkImageData()
partition_img.SetDimensions(nbx + 1, nby + 1, 1)
partition_img.SetOrigin(origin[0], origin[1], 0.0)
partition_img.SetSpacing(spacing[0] * BLOCK, spacing[1] * BLOCK, 1.0)
carr = ns.numpy_to_vtk(block_class.ravel(), deep=True)
carr.SetName("EigenvectorPartition")
partition_img.GetCellData().AddArray(carr)

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("/workspace/Ocean_partition.vti")
writer.SetInputData(partition_img)
writer.Write()

# ---------------------------------------------------------------------------
# 3. Degenerate point extraction (E = F = 0) on triangulated cells,
#    classified as wedge / trisector via the sign of the (E,F) Jacobian.
# ---------------------------------------------------------------------------
def bary_root(p0, p1, p2, f0, f1, f2, g0, g1, g2):
    """Find (s,t) in triangle (barycentric on edges from p0) solving
    f(s,t)=0, g(s,t)=0 for f,g linear on the triangle. Returns None if
    the root is outside the triangle."""
    # f(s,t) = f0 + s*(f1-f0) + t*(f2-f0), same for g
    a11, a12 = (f1 - f0), (f2 - f0)
    a21, a22 = (g1 - g0), (g2 - g0)
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-14:
        return None
    s = (-f0 * a22 + g0 * a12) / det
    t = (-g0 * a11 + f0 * a21) / det
    if s < -1e-9 or t < -1e-9 or (s + t) > 1 + 1e-9:
        return None
    return s, t

points = vtk.vtkPoints()
ptypes = []  # 0 = wedge, 1 = trisector

def jacobian_sign(p0, p1, p2, e0, e1, e2, f0, f1, f2):
    # linear map: solve for gradients of E and F over the triangle
    M = np.array([[p1[0] - p0[0], p1[1] - p0[1]],
                  [p2[0] - p0[0], p2[1] - p0[1]]])
    rhsE = np.array([e1 - e0, e2 - e0])
    rhsF = np.array([f1 - f0, f2 - f0])
    try:
        gradE = np.linalg.solve(M, rhsE)
        gradF = np.linalg.solve(M, rhsF)
    except np.linalg.LinAlgError:
        return 0.0
    return gradE[0] * gradF[1] - gradE[1] * gradF[0]

X0, Y0 = origin[0], origin[1]
sx, sy = spacing[0], spacing[1]

for j in range(ny - 1):
    for i in range(nx - 1):
        # cell corners (00, 10, 11, 01) -> two triangles (00,10,11) and (00,11,01)
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        tris = [(0, 1, 2), (0, 2, 3)]
        for (ia, ib, ic) in tris:
            ca, cb, cc = corners[ia], corners[ib], corners[ic]
            p0 = (X0 + ca[0] * sx, Y0 + ca[1] * sy)
            p1 = (X0 + cb[0] * sx, Y0 + cb[1] * sy)
            p2 = (X0 + cc[0] * sx, Y0 + cc[1] * sy)
            e0, e1, e2 = E[ca[1], ca[0]], E[cb[1], cb[0]], E[cc[1], cc[0]]
            f0, f1, f2 = F[ca[1], ca[0]], F[cb[1], cb[0]], F[cc[1], cc[0]]
            root = bary_root(p0, p1, p2, e0, e1, e2, f0, f1, f2)
            if root is None:
                continue
            s, t = root
            x = p0[0] + s * (p1[0] - p0[0]) + t * (p2[0] - p0[0])
            y = p0[1] + s * (p1[1] - p0[1]) + t * (p2[1] - p0[1])
            jac = jacobian_sign(p0, p1, p2, e0, e1, e2, f0, f1, f2)
            points.InsertNextPoint(x, y, 0.0)
            ptypes.append(0 if jac > 0 else 1)  # 0=wedge, 1=trisector

poly = vtk.vtkPolyData()
poly.SetPoints(points)
type_arr = ns.numpy_to_vtk(np.array(ptypes, dtype=np.uint8), deep=True)
type_arr.SetName("DegenerateType")  # 0 = wedge, 1 = trisector
poly.GetPointData().AddArray(type_arr)

verts = vtk.vtkCellArray()
for idx in range(points.GetNumberOfPoints()):
    verts.InsertNextCell(1, [idx])
poly.SetVerts(verts)

pwriter = vtk.vtkXMLPolyDataWriter()
pwriter.SetFileName("/workspace/Ocean_degenerate_points.vtp")
pwriter.SetInputData(poly)
pwriter.Write()

n_wedge = ptypes.count(0)
n_tri = ptypes.count(1)
print("Degenerate points found: %d wedges, %d trisectors" % (n_wedge, n_tri))

# ---------------------------------------------------------------------------
# 4. Visualize in ParaView: partition raster (colored squares) +
#    degenerate points as radius-1 spheres, pink = trisector, white = wedge
# ---------------------------------------------------------------------------
partReader = XMLImageDataReader(FileName=["/workspace/Ocean_partition.vti"])
partReader.UpdatePipeline()
partDisplay = Show(partReader)
partDisplay.Representation = "Surface"
ColorBy(partDisplay, ("CELLS", "EigenvectorPartition"))
partLUT = GetColorTransferFunction("EigenvectorPartition")
partLUT.InterpretValuesAsCategories = 1
partLUT.AnnotationsInitialized = 1
partLUT.Annotations = ["0", "Complex (no real eigenvectors)", "1", "Real (two real eigenvectors)"]
partLUT.IndexedColors = [0.55, 0.55, 0.55,   # complex domain -> gray
                          0.25, 0.45, 0.75]  # real domain -> blue
partDisplay.SetScalarBarVisibility(GetActiveView(), True)

degReader = XMLPolyDataReader(FileName=["/workspace/Ocean_degenerate_points.vtp"])
degReader.UpdatePipeline()

glyph = Glyph(Input=degReader, GlyphType="Sphere")
glyph.GlyphType.Radius = GLYPH_RADIUS
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.ScaleFactor = 1.0
glyph.GlyphMode = "All Points"

glyphDisplay = Show(glyph)
glyphDisplay.Representation = "Surface"
ColorBy(glyphDisplay, ("POINTS", "DegenerateType"))
degLUT = GetColorTransferFunction("DegenerateType")
degLUT.InterpretValuesAsCategories = 1
degLUT.AnnotationsInitialized = 1
degLUT.Annotations = ["0", "Wedge", "1", "Trisector"]
degLUT.IndexedColors = [1.0, 1.0, 1.0,   # wedge -> white
                         1.0, 0.75, 0.8]  # trisector -> pink

view = GetActiveView()
view.OrientationAxesVisibility = 0
ResetCamera()
view.ViewSize = [900, 900]
Render()
SaveScreenshot("/workspace/Ocean_eigenvector_partition.png", view, ImageResolution=[900, 900])

print("Saved:")
print(" /workspace/Ocean_partition.vti")
print(" /workspace/Ocean_degenerate_points.vtp")
print(" /workspace/Ocean_eigenvector_partition.png")
