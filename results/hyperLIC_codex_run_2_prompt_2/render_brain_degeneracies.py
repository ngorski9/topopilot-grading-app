"""Render a 2-D symmetric tensor field and its PL degeneracies.

For a tensor [[A,B],[B,D]], degeneracies are the simultaneous zeros of
(A-D, 2B).  On each triangle of the image-grid triangulation both functions
are linear, so their common zero is found exactly with a 2x2 solve.
"""
import math
import os
import vtk

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "brain.vti")
OUTPUT = os.path.join(HERE, "brain_degeneracies.png")

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
image = reader.GetOutput()
pd = image.GetPointData()
a, b, d = (pd.GetArray(n) for n in ("A", "B", "D"))
if not all((a, b, d)):
    raise RuntimeError("brain.vti must contain point arrays A, B, and D")

dims = image.GetDimensions()
nx, ny = dims[0], dims[1]

# Create an anisotropy image for the field backdrop.
ani = vtk.vtkDoubleArray(); ani.SetName("Tensor anisotropy")
ani.SetNumberOfTuples(nx * ny)
for pid in range(nx * ny):
    ani.SetValue(pid, math.hypot(a.GetValue(pid)-d.GetValue(pid), 2*b.GetValue(pid)))
field_image = vtk.vtkImageData(); field_image.DeepCopy(image)
field_image.GetPointData().AddArray(ani)
field_image.GetPointData().SetScalars(ani)

# Extract all isolated roots from the fixed two-triangle subdivision of cells.
# Duplicates on shared edges are coalesced at numerical tolerance.
roots = []
def val(i, j):
    q = j * nx + i
    return (a.GetValue(q) - d.GetValue(q), 2.0*b.GetValue(q))
def add_triangle(p0, p1, p2):
    f0, f1, f2 = val(*p0), val(*p1), val(*p2)
    m00, m01 = f1[0]-f0[0], f2[0]-f0[0]
    m10, m11 = f1[1]-f0[1], f2[1]-f0[1]
    det = m00*m11 - m01*m10
    if abs(det) < 1e-12:
        return
    u = (-f0[0]*m11 + m01*f0[1]) / det
    v = (-m00*f0[1] + f0[0]*m10) / det
    eps = 1e-8
    if u >= -eps and v >= -eps and u+v <= 1+eps:
        x = p0[0] + u*(p1[0]-p0[0]) + v*(p2[0]-p0[0])
        y = p0[1] + u*(p1[1]-p0[1]) + v*(p2[1]-p0[1])
        # Positive director index is a wedge, negative is a trisector.
        kind = "wedge" if det > 0 else "trisector"
        for old in roots:
            if (x-old[0])**2 + (y-old[1])**2 < 1e-10:
                return
        roots.append((x, y, kind))

for j in range(ny-1):
    for i in range(nx-1):
        add_triangle((i,j), (i+1,j), (i+1,j+1))
        add_triangle((i,j), (i+1,j+1), (i,j+1))

def make_points(which):
    pts = vtk.vtkPoints()
    for x, y, kind in roots:
        if kind == which:
            pts.InsertNextPoint(x, y, 0.35)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for i in range(pts.GetNumberOfPoints()):
        verts.InsertNextCell(1); verts.InsertCellPoint(i)
    poly.SetVerts(verts)
    return poly

# Tensor glyphs: short major-eigenvector strokes reveal the directional field.
glyph_pts = vtk.vtkPoints(); glyph_lines = vtk.vtkCellArray()
for j in range(1, ny-1, 4):
    for i in range(1, nx-1, 4):
        pid = j*nx+i; aa, bb, dd = a.GetValue(pid), b.GetValue(pid), d.GetValue(pid)
        strength = math.hypot(aa-dd, 2*bb)
        if strength < 0.025:
            continue
        theta = 0.5 * math.atan2(2*bb, aa-dd)
        length = 1.25
        q0 = glyph_pts.InsertNextPoint(i-length*math.cos(theta), j-length*math.sin(theta), 0.12)
        q1 = glyph_pts.InsertNextPoint(i+length*math.cos(theta), j+length*math.sin(theta), 0.12)
        glyph_lines.InsertNextCell(2); glyph_lines.InsertCellPoint(q0); glyph_lines.InsertCellPoint(q1)
glyphs = vtk.vtkPolyData(); glyphs.SetPoints(glyph_pts); glyphs.SetLines(glyph_lines)

# Headless raster export.  VTK reads and processes the VTI data above; use
# matplotlib for a portable image backend when no OpenGL display is present.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8.5), dpi=100, facecolor="#09090f")
ax.set_facecolor("#09090f")
aniso_grid = np.array([ani.GetValue(i) for i in range(nx*ny)]).reshape(ny, nx)
ax.imshow(aniso_grid, origin="lower", extent=(-.5, nx-.5, -.5, ny-.5),
          cmap="viridis", interpolation="bilinear", vmin=0, vmax=aniso_grid.max())
# Directional strokes provide the tensor-field rendering.
for line_id in range(glyph_lines.GetNumberOfCells()):
    cell = glyph_lines.GetCell(line_id)
    p, q = glyph_pts.GetPoint(cell.GetPointId(0)), glyph_pts.GetPoint(cell.GetPointId(1))
    ax.plot((p[0], q[0]), (p[1], q[1]), color="#08080c", alpha=.62, lw=.55, solid_capstyle="round", zorder=2)
for kind, color, edge in (("trisector", "#ff5a9a", "#ffb6d0"), ("wedge", "#ffffff", "#26313d")):
    circles = [Circle((x, y), radius=1.0) for x, y, typ in roots if typ == kind]
    ax.add_collection(PatchCollection(circles, facecolor=color, edgecolor=edge, linewidth=.35, zorder=3))
ax.set_xlim(-.5, nx-.5); ax.set_ylim(-.5, ny-.5); ax.set_aspect("equal"); ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUTPUT, dpi=100, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
plt.close(fig)
print(f"Wrote {OUTPUT}: {sum(r[2]=='trisector' for r in roots)} trisectors, {sum(r[2]=='wedge' for r in roots)} wedges")
