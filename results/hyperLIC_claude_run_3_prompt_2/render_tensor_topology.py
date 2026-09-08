import vtk
import numpy as np
from vtk.util import numpy_support as ns

# ---------------------------------------------------------------
# 1. Load the piecewise-linear symmetric tensor field brain.vti
# ---------------------------------------------------------------
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('/workspace/brain.vti')
reader.Update()
img = reader.GetOutput()

dims = img.GetDimensions()
nx, ny, nz = dims
assert nz == 1, "expected a 2D field"

pd = img.GetPointData()
A = ns.vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = ns.vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = ns.vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

origin = img.GetOrigin()
spacing = img.GetSpacing()

def coord(i, j):
    return (origin[0] + i * spacing[0], origin[1] + j * spacing[1])

# Deviator components: degenerate points are common zeros of E and F
E = A - D
F = 2.0 * B

# ---------------------------------------------------------------
# 2. Triangulate the grid (2 triangles per quad, CCW) and locate
#    degenerate points as the common zero of (E,F) inside each
#    triangle (Delmarcelle & Hesselink, 1994).
# ---------------------------------------------------------------
degenerate_points = []  # (x, y, kind)  kind: 'wedge' or 'trisector'

def process_triangle(p0, p1, p2, e0, e1, e2, f0, f1, f2):
    # solve for barycentric l0,l1 (l2 = 1-l0-l1) such that
    # l0*E0+l1*E1+l2*E2 = 0 ; l0*F0+l1*F1+l2*F2 = 0
    m00 = e0 - e2
    m01 = e1 - e2
    m10 = f0 - f2
    m11 = f1 - f2
    det = m00 * m11 - m01 * m10
    if det == 0.0:
        return
    rhs0 = -e2
    rhs1 = -f2
    l0 = (m11 * rhs0 - m01 * rhs1) / det
    l1 = (-m10 * rhs0 + m00 * rhs1) / det
    l2 = 1.0 - l0 - l1
    eps = -1e-9
    if l0 >= eps and l1 >= eps and l2 >= eps:
        x = l0 * p0[0] + l1 * p1[0] + l2 * p2[0]
        y = l0 * p0[1] + l1 * p1[1] + l2 * p2[1]
        # sign of det (with CCW-oriented triangle) classifies the point:
        # delta > 0 -> wedge ; delta < 0 -> trisector
        kind = 'wedge' if det > 0 else 'trisector'
        degenerate_points.append((x, y, kind))

for j in range(ny - 1):
    for i in range(nx - 1):
        p00, p10, p11, p01 = coord(i, j), coord(i + 1, j), coord(i + 1, j + 1), coord(i, j + 1)
        e00, e10, e11, e01 = E[j, i], E[j, i + 1], E[j + 1, i + 1], E[j + 1, i]
        f00, f10, f11, f01 = F[j, i], F[j, i + 1], F[j + 1, i + 1], F[j + 1, i]
        # triangle 1: p00, p10, p11 (CCW)
        process_triangle(p00, p10, p11, e00, e10, e11, f00, f10, f11)
        # triangle 2: p00, p11, p01 (CCW)
        process_triangle(p00, p11, p01, e00, e11, e01, f00, f11, f01)

print(f"Detected {len(degenerate_points)} degenerate points "
      f"({sum(1 for *_, k in degenerate_points if k=='wedge')} wedges, "
      f"{sum(1 for *_, k in degenerate_points if k=='trisector')} trisectors)")

# ---------------------------------------------------------------
# 3. Build a vtkTensorGlyph representation of the field itself
#    (subsampled so the picture stays legible), embedding the 2x2
#    symmetric tensor [[A,B],[B,D]] into a 3x3 tensor.
# ---------------------------------------------------------------
stride = 2
sub_points = vtk.vtkPoints()
tensors = vtk.vtkDoubleArray()
tensors.SetNumberOfComponents(9)
trace_scalars = vtk.vtkDoubleArray()
trace_scalars.SetName('trace')

for j in range(0, ny, stride):
    for i in range(0, nx, stride):
        x, y = coord(i, j)
        sub_points.InsertNextPoint(x, y, 0.0)
        a, b, d = A[j, i], B[j, i], D[j, i]
        tensors.InsertNextTuple9(a, b, 0, b, d, 0, 0, 0, 0)
        trace_scalars.InsertNextValue(a + d)

tensor_field = vtk.vtkPolyData()
tensor_field.SetPoints(sub_points)
tensor_field.GetPointData().SetTensors(tensors)
tensor_field.GetPointData().SetScalars(trace_scalars)

ellipse = vtk.vtkSphereSource()  # base glyph flattened by tensor scaling into an ellipse
ellipse.SetThetaResolution(12)
ellipse.SetPhiResolution(12)
ellipse.SetRadius(1.0)

max_eig = max(abs(A).max(), abs(D).max(), abs(B).max())
target_size = 0.9 * stride
scale_factor = target_size / max_eig if max_eig > 0 else 1.0

glyph = vtk.vtkTensorGlyph()
glyph.SetInputData(tensor_field)
glyph.SetSourceConnection(ellipse.GetOutputPort())
glyph.ColorGlyphsOn()
glyph.SetColorModeToScalars()
glyph.ThreeGlyphsOff()
glyph.ExtractEigenvaluesOn()
glyph.SetScaleFactor(scale_factor)
glyph.ClampScalingOff()
glyph.Update()

glyph_mapper = vtk.vtkPolyDataMapper()
glyph_mapper.SetInputConnection(glyph.GetOutputPort())
glyph_mapper.ScalarVisibilityOn()
glyph_mapper.SetScalarRange(trace_scalars.GetRange())
glyph_mapper.SetLookupTable(vtk.vtkLookupTable())
glyph_mapper.GetLookupTable().SetHueRange(0.667, 0.0)  # blue -> red
glyph_mapper.GetLookupTable().Build()

glyph_actor = vtk.vtkActor()
glyph_actor.SetMapper(glyph_mapper)

# ---------------------------------------------------------------
# 4. Degenerate points as spheres of radius 1: pink = trisector,
#    white = wedge.
# ---------------------------------------------------------------
def make_sphere_actor(pts, color):
    points = vtk.vtkPoints()
    for x, y, _ in pts:
        points.InsertNextPoint(x, y, 0.01)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)

    g = vtk.vtkGlyph3D()
    g.SetInputData(poly)
    g.SetSourceConnection(sphere.GetOutputPort())
    g.SetScaleModeToDataScalingOff()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(g.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    return actor

wedges = [p for p in degenerate_points if p[2] == 'wedge']
trisectors = [p for p in degenerate_points if p[2] == 'trisector']

wedge_actor = make_sphere_actor(wedges, (1.0, 1.0, 1.0))          # white
trisector_actor = make_sphere_actor(trisectors, (1.0, 0.41, 0.71))  # pink

# ---------------------------------------------------------------
# 5. Render
# ---------------------------------------------------------------
renderer = vtk.vtkRenderer()
renderer.AddActor(glyph_actor)
renderer.AddActor(wedge_actor)
renderer.AddActor(trisector_actor)
renderer.SetBackground(0.08, 0.08, 0.1)

render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1600, 1000)

renderer.ResetCamera()
renderer.GetActiveCamera().Zoom(1.05)

render_window.Render()

w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(render_window)
w2if.SetInputBufferTypeToRGB()
w2if.ReadFrontBufferOff()
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName('/workspace/brain_tensor_topology.png')
writer.SetInputConnection(w2if.GetOutputPort())
writer.Write()

print("Saved /workspace/brain_tensor_topology.png")
