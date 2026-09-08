from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import numpy as np

INPUT = "/workspace/brain.vti"
OUT_PNG = "/workspace/brain_tensor_degenerate_points.png"
OUT_VTP = "/workspace/brain_degenerate_points.vtp"
OUT_STATE = "/workspace/brain_tensor_degenerate_points.pvsm"

# Read the three independent components of the symmetric 2-D tensor.
reader = XMLImageDataReader(FileName=[INPUT])
reader.UpdatePipeline()
image = reader.GetClientSideObject().GetOutputDataObject(0)
pd = image.GetPointData()
a = vtk_to_numpy(pd.GetArray("A"))
b = vtk_to_numpy(pd.GetArray("B"))
d = vtk_to_numpy(pd.GetArray("D"))
dims = image.GetDimensions()
nx, ny = dims[0], dims[1]

# A tensor is degenerate where (A-D, 2B) = (0, 0).  On each triangle this
# pair is linear, so solve its 2x2 barycentric system exactly.
q = a - d
r = 2.0 * b
trace = a + d
threshold = trace.max() * 1.0e-5
points = []
classes = []
eps = 1.e-12
for j in range(ny - 1):
    for i in range(nx - 1):
        ids = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1,
               (j + 1) * nx + i]
        # Fixed diagonal produces a piecewise-linear field.
        for tri in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
            if min(trace[k] for k in tri) <= threshold:
                continue
            mat = np.array([[q[tri[1]] - q[tri[0]], q[tri[2]] - q[tri[0]]],
                            [r[tri[1]] - r[tri[0]], r[tri[2]] - r[tri[0]]]])
            rhs = -np.array([q[tri[0]], r[tri[0]]])
            det = np.linalg.det(mat)
            if abs(det) < eps:
                continue
            u, v = np.linalg.solve(mat, rhs)
            w = 1.0 - u - v
            if u >= -eps and v >= -eps and w >= -eps:
                xy = np.array([[image.GetPoint(k)[0], image.GetPoint(k)[1]] for k in tri])
                p = w * xy[0] + u * xy[1] + v * xy[2]
                # The sign of det(A-D,2B) is the doubled line-field index:
                # positive is a wedge (+1/2), negative a trisector (-1/2).
                points.append((p[0], p[1], 0.15))
                classes.append(1 if det > 0 else 0)  # 1=wedge, 0=trisector

# Remove the duplicate generated when a zero lies on a triangulation edge.
unique = []
for p, c in zip(points, classes):
    if not any((p[0]-old[0])**2 + (p[1]-old[1])**2 < 1.e-10 for old, _ in unique):
        unique.append((p, c))
points, classes = zip(*unique) if unique else ([], [])

poly = vtk.vtkPolyData()
vpts = vtk.vtkPoints()
for p in points: vpts.InsertNextPoint(p)
poly.SetPoints(vpts)
verts = vtk.vtkCellArray()
for k in range(len(points)):
    verts.InsertNextCell(1); verts.InsertCellPoint(k)
poly.SetVerts(verts)
kind = vtk.vtkIntArray(); kind.SetName("DegenerateType")
for c in classes: kind.InsertNextValue(c)
poly.GetPointData().AddArray(kind)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(OUT_VTP); writer.SetInputData(poly); writer.Write()

# Field backdrop: tensor magnitude contours retain the anatomy without
# conflating the requested point colors with the field colormap.
mag = np.sqrt(a*a + 2*b*b + d*d)
mag_arr = numpy_to_vtk(mag, deep=True); mag_arr.SetName("TensorMagnitude")
image.GetPointData().AddArray(mag_arr)
writer_i = vtk.vtkXMLImageDataWriter(); writer_i.SetFileName("/workspace/brain_tensor_magnitude.vti"); writer_i.SetInputData(image); writer_i.Write()

field = XMLImageDataReader(FileName=["/workspace/brain_tensor_magnitude.vti"])
contour = Contour(registrationName="Brain tensor field", Input=field)
contour.ContourBy = ['POINTS', 'TensorMagnitude']
contour.Isosurfaces = [float(mag.max() * 0.04)]
field_display = Show(contour)
field_display.DiffuseColor = [0.32, 0.36, 0.43]
field_display.Opacity = 0.88

pts_source = XMLPolyDataReader(FileName=[OUT_VTP])
tris = Threshold(registrationName="Trisectors", Input=pts_source)
tris.Scalars = ['POINTS', 'DegenerateType']; tris.LowerThreshold = 0; tris.UpperThreshold = 0
wedges = Threshold(registrationName="Wedges", Input=pts_source)
wedges.Scalars = ['POINTS', 'DegenerateType']; wedges.LowerThreshold = 1; wedges.UpperThreshold = 1

# Unit-radius spheres are used for both classifications.
for src, name, color in ((tris, 'Trisectors (radius 1)', [1.0, 0.35, 0.65]),
                         (wedges, 'Wedges (radius 1)', [1.0, 1.0, 1.0])):
    sphere = Glyph(registrationName=name, Input=src, GlyphType='Sphere')
    sphere.GlyphType.Radius = 1.0
    sphere.ScaleArray = ['POINTS', 'No scale array']
    sphere.ScaleFactor = 1.0
    display = Show(sphere)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.35

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1280, 800]
view.Background = [0.045, 0.055, 0.08]
view.InteractionMode = '2D'
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
Render(); view.ResetCamera(); view.CameraParallelScale *= 1.08
Render()
SaveScreenshot(OUT_PNG, view, ImageResolution=[1280, 800])
SaveState(OUT_STATE)
print('degenerate points:', len(points), 'trisectors:', sum(c == 0 for c in classes), 'wedges:', sum(c == 1 for c in classes))
print(OUT_PNG)
