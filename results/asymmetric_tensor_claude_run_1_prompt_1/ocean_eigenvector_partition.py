"""
Eigenvector-partition visualization of the asymmetric 2x2 tensor field stored
in Ocean.vti (components A, B, C, D -> T = [[A, B], [C, D]]).

Pipeline
--------
1. Read Ocean.vti (a 101x101 vtkImageData, spacing 1, so the domain is a
   100x100 pixel square) and pull out A, B, C, D as numpy arrays.
2. Degenerate points are found with the classical Delmarcelle & Hesselink
   (1994) construction applied to the symmetric deviatoric part of the
   tensor:
        E = (A - D) / 2 ,   F = (B + C) / 2
   A degenerate point is a zero of the vector field (E, F). Within every
   grid cell, E and F are bilinearly interpolated from the four corner
   values and the zero(s) of the resulting 2x2 bilinear system are found
   in closed form (reduces to a quadratic). The type of each degenerate
   point is given by the sign of the local Jacobian
        delta = dE/ds * dF/dt - dE/dt * dF/ds
   delta < 0  -> trisector (index -1/2)
   delta > 0  -> wedge     (index +1/2)
3. The eigenvector partition is drawn on a coarse grid with one square per
   10x10 pixel block (i.e. a "resolution of 10 pixels per square"): the
   symmetric part of the tensor is diagonalized at the centre of every
   square, the square is colored by whether the tensor is degenerate-free
   there, and a cross glyph shows the two (perpendicular) eigenvector
   directions.
4. Degenerate points are rendered as spheres of radius 1: pink for
   trisectors, white for wedges.

Output: renders an offscreen image to Ocean_eigenvector_partition.png
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

INPUT_FILE = "Ocean.vti"
OUTPUT_IMAGE = "Ocean_eigenvector_partition.png"
SQUARE = 10.0   # pixels per square for the eigenvector-partition display
GLYPH_RADIUS = 1.0

# ---------------------------------------------------------------------
# 1. Read the data
# ---------------------------------------------------------------------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT_FILE)
reader.Update()
img = reader.GetOutput()
dims = img.GetDimensions()
nx, ny = dims[0], dims[1]
ox, oy, _ = img.GetOrigin()
sx, sy, _ = img.GetSpacing()

pd = img.GetPointData()


def arr(name):
    return vtk_to_numpy(pd.GetArray(name)).reshape(ny, nx)


A = arr("A")
B = arr("B")
C = arr("C")
D = arr("D")

E = 0.5 * (A - D)
F = 0.5 * (B + C)


def xcoord(i):
    return ox + i * sx


def ycoord(j):
    return oy + j * sy


# ---------------------------------------------------------------------
# 2. Degenerate point extraction (cell-wise bilinear zero of (E, F))
# ---------------------------------------------------------------------
degenerate_points = []  # (x, y, kind) kind in {"trisector", "wedge"}

for j in range(ny - 1):
    for i in range(nx - 1):
        E00, E10, E01, E11 = E[j, i], E[j, i + 1], E[j + 1, i], E[j + 1, i + 1]
        F00, F10, F01, F11 = F[j, i], F[j, i + 1], F[j + 1, i], F[j + 1, i + 1]

        e0, e1, e2 = E00, E10 - E00, E01 - E00
        e3 = E00 - E10 - E01 + E11
        f0, f1, f2 = F00, F10 - F00, F01 - F00
        f3 = F00 - F10 - F01 + F11

        Aq = f2 * e3 - e2 * f3
        Bq = f0 * e3 + f2 * e1 - e0 * f3 - e2 * f1
        Cq = f0 * e1 - e0 * f1

        roots = []
        if abs(Aq) < 1e-14:
            if abs(Bq) > 1e-14:
                roots = [-Cq / Bq]
        else:
            disc = Bq * Bq - 4 * Aq * Cq
            if disc >= 0:
                sq = np.sqrt(disc)
                roots = [(-Bq + sq) / (2 * Aq), (-Bq - sq) / (2 * Aq)]

        for t in roots:
            if t < -1e-9 or t > 1 + 1e-9:
                continue
            denom = e1 + e3 * t
            if abs(denom) < 1e-14:
                continue
            s = -(e0 + e2 * t) / denom
            if s < -1e-9 or s > 1 + 1e-9:
                continue

            Es = e1 + e3 * t
            Et = e2 + e3 * s
            Fs = f1 + f3 * t
            Ft = f2 + f3 * s
            delta = Es * Ft - Et * Fs

            x = xcoord(i + s)
            y = ycoord(j + t)
            kind = "trisector" if delta < 0 else "wedge"
            degenerate_points.append((x, y, kind))

# de-duplicate points that got found from both quadratic roots landing on
# the same spot / shared edges between neighbouring cells
uniq = []
for p in degenerate_points:
    if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
        uniq.append(p)
degenerate_points = uniq

n_tri = sum(1 for p in degenerate_points if p[2] == "trisector")
n_wed = sum(1 for p in degenerate_points if p[2] == "wedge")
print(f"Found {len(degenerate_points)} degenerate points "
      f"({n_tri} trisectors, {n_wed} wedges)")

# ---------------------------------------------------------------------
# 3. Eigenvector partition: coarse grid, one square per SQUARE pixels
# ---------------------------------------------------------------------


def bilinear(field, x, y):
    fx = (x - ox) / sx
    fy = (y - oy) / sy
    i0 = int(np.clip(np.floor(fx), 0, nx - 2))
    j0 = int(np.clip(np.floor(fy), 0, ny - 2))
    s = fx - i0
    t = fy - j0
    v00, v10 = field[j0, i0], field[j0, i0 + 1]
    v01, v11 = field[j0 + 1, i0], field[j0 + 1, i0 + 1]
    return (v00 * (1 - s) * (1 - t) + v10 * s * (1 - t)
            + v01 * (1 - s) * t + v11 * s * t)


x_min, x_max = xcoord(0), xcoord(nx - 1)
y_min, y_max = ycoord(0), ycoord(ny - 1)

n_sq_x = int(round((x_max - x_min) / SQUARE))
n_sq_y = int(round((y_max - y_min) / SQUARE))

partition_polydata = vtk.vtkAppendPolyData()
square_colors = vtk.vtkUnsignedCharArray()
square_colors.SetNumberOfComponents(3)
square_colors.SetName("partition_color")

square_polys = vtk.vtkAppendPolyData()

glyph_len = 0.42 * SQUARE  # half-length of each eigenvector arm

for jj in range(n_sq_y):
    for ii in range(n_sq_x):
        cx = x_min + (ii + 0.5) * SQUARE
        cy = y_min + (jj + 0.5) * SQUARE

        Ac = bilinear(A, cx, cy)
        Bc = bilinear(B, cx, cy)
        Cc = bilinear(C, cx, cy)
        Dc = bilinear(D, cx, cy)

        # symmetric part -> always real eigenvalues/eigenvectors
        Sxx, Sxy, Syy = Ac, 0.5 * (Bc + Cc), Dc
        M = np.array([[Sxx, Sxy], [Sxy, Syy]])
        evals, evecs = np.linalg.eigh(M)

        # background square colored by partition class: which eigenvalue
        # (major = largest magnitude) currently dominates the x axis vs y axis
        major = evecs[:, np.argmax(np.abs(evals))]
        # classify by dominant orientation quadrant of the major eigenvector
        angle = np.arctan2(major[1], major[0]) % np.pi
        is_x_dominant = angle < np.pi / 4 or angle > 3 * np.pi / 4

        square = vtk.vtkPlaneSource()
        square.SetOrigin(cx - SQUARE / 2, cy - SQUARE / 2, -0.02)
        square.SetPoint1(cx + SQUARE / 2, cy - SQUARE / 2, -0.02)
        square.SetPoint2(cx - SQUARE / 2, cy + SQUARE / 2, -0.02)
        square.Update()
        sp = square.GetOutput()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("colors")
        col = (222, 235, 247) if is_x_dominant else (204, 224, 205)
        for _ in range(sp.GetNumberOfPoints()):
            colors.InsertNextTuple3(*col)
        sp.GetPointData().SetScalars(colors)
        square_polys.AddInputData(sp)

        # eigenvector cross glyph
        for k in range(2):
            v = evecs[:, k]
            line = vtk.vtkLineSource()
            line.SetPoint1(cx - v[0] * glyph_len, cy - v[1] * glyph_len, 0.0)
            line.SetPoint2(cx + v[0] * glyph_len, cy + v[1] * glyph_len, 0.0)
            line.Update()
            partition_polydata.AddInputData(line.GetOutput())

square_polys.Update()
partition_polydata.Update()

square_mapper = vtk.vtkPolyDataMapper()
square_mapper.SetInputConnection(square_polys.GetOutputPort())
square_actor = vtk.vtkActor()
square_actor.SetMapper(square_mapper)

lines_mapper = vtk.vtkPolyDataMapper()
lines_mapper.SetInputConnection(partition_polydata.GetOutputPort())
lines_actor = vtk.vtkActor()
lines_actor.SetMapper(lines_mapper)
lines_actor.GetProperty().SetColor(0.1, 0.1, 0.1)
lines_actor.GetProperty().SetLineWidth(1.5)

# ---------------------------------------------------------------------
# 4. Degenerate points as colored spheres (radius 1)
# ---------------------------------------------------------------------
tri_append = vtk.vtkAppendPolyData()
wed_append = vtk.vtkAppendPolyData()
any_tri = any_wed = False

for x, y, kind in degenerate_points:
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(x, y, 0.05)
    sphere.SetRadius(GLYPH_RADIUS)
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)
    sphere.Update()
    if kind == "trisector":
        tri_append.AddInputData(sphere.GetOutput())
        any_tri = True
    else:
        wed_append.AddInputData(sphere.GetOutput())
        any_wed = True

actors_extra = []
if any_tri:
    tri_append.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(tri_append.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(1.0, 0.41, 0.71)  # pink
    actors_extra.append(a)
if any_wed:
    wed_append.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(wed_append.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(1.0, 1.0, 1.0)  # white
    a.GetProperty().SetAmbient(0.6)
    actors_extra.append(a)

# ---------------------------------------------------------------------
# 5. Render
# ---------------------------------------------------------------------
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.15, 0.15, 0.18)
renderer.AddActor(square_actor)
renderer.AddActor(lines_actor)
for a in actors_extra:
    renderer.AddActor(a)

render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1000, 1000)

renderer.ResetCamera()
cam = renderer.GetActiveCamera()
cam.ParallelProjectionOn()
cam.SetParallelScale((y_max - y_min) / 2 * 1.05)

render_window.Render()

w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(render_window)
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(OUTPUT_IMAGE)
writer.SetInputConnection(w2if.GetOutputPort())
writer.Write()

print(f"Wrote {OUTPUT_IMAGE}")
