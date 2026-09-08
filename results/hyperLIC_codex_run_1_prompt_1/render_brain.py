from paraview.simple import *
from paraview import servermanager
import vtk
import math

input_file = '/workspace/brain.vti'
output_file = '/workspace/brain_degenerate_points.png'

# Read through VTK so the locations can be computed exactly in each PL triangle.
r = vtk.vtkXMLImageDataReader()
r.SetFileName(input_file)
r.Update()
img = r.GetOutput()
pd = img.GetPointData()
a, b, d = (pd.GetArray(n) for n in ('A', 'B', 'D'))
extent = img.GetExtent(); origin = img.GetOrigin(); spacing = img.GetSpacing()
nx = extent[1] - extent[0] + 1; ny = extent[3] - extent[2] + 1

def pid(i, j): return (j - extent[2]) * nx + (i - extent[0])
def xy(i, j): return (origin[0] + i * spacing[0], origin[1] + j * spacing[1])

# Degeneracy means the traceless part (A-D, 2B) vanishes.  On every
# piecewise-linear triangle this is a 2x2 linear solve.  The determinant
# gives the local tensor-line index: +1/2 wedges, -1/2 trisectors.
found = {}
for j in range(extent[2], extent[3]):
  for i in range(extent[0], extent[1]):
    corners = [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]
    for tri in ((0,1,2), (0,2,3)):
      vv = [corners[k] for k in tri]
      q = [(a.GetTuple1(pid(x,y))-d.GetTuple1(pid(x,y)), 2*b.GetTuple1(pid(x,y))) for x,y in vv]
      m00, m01 = q[1][0]-q[0][0], q[2][0]-q[0][0]
      m10, m11 = q[1][1]-q[0][1], q[2][1]-q[0][1]
      det = m00*m11-m01*m10
      if abs(det) < 1e-12: continue
      u = ((-q[0][0])*m11 - m01*(-q[0][1])) / det
      v = (m00*(-q[0][1]) - (-q[0][0])*m10) / det
      w = 1-u-v
      eps = 1e-8
      if u >= -eps and v >= -eps and w >= -eps:
        pts = [xy(*p) for p in vv]
        x = w*pts[0][0]+u*pts[1][0]+v*pts[2][0]
        y = w*pts[0][1]+u*pts[1][1]+v*pts[2][1]
        key = (round(x, 6), round(y, 6))
        found[key] = 1 if det > 0 else -1

wedge_pts, tri_pts = [], []
for (x,y), sign in found.items():
  (wedge_pts if sign > 0 else tri_pts).append((x,y,0.15))
print('Degenerate points:', len(found), 'wedges:', len(wedge_pts), 'trisectors:', len(tri_pts))

def make_circles(points, name):
  poly = vtk.vtkPolyData(); outpts = vtk.vtkPoints(); cells = vtk.vtkCellArray()
  # Geometric radius is exactly one grid-length unit.
  for x,y,z in points:
    start = outpts.GetNumberOfPoints(); n = 32
    for k in range(n):
      ang = 2*math.pi*k/n
      outpts.InsertNextPoint(x + math.cos(ang), y + math.sin(ang), z)
    cell = vtk.vtkPolygon(); cell.GetPointIds().SetNumberOfIds(n)
    for k in range(n): cell.GetPointIds().SetId(k, start+k)
    cells.InsertNextCell(cell)
  poly.SetPoints(outpts); poly.SetPolys(cells)
  src = TrivialProducer(registrationName=name); src.GetClientSideObject().SetOutput(poly)
  return src

# A scalar view makes the loaded tensor field visible while avoiding an
# arbitrary choice of tensor glyph scale.  It is the Frobenius magnitude.
calc = Calculator(Input=XMLImageDataReader(registrationName='brain tensor field', FileName=[input_file]))
calc.ResultArrayName = 'Tensor Frobenius magnitude'
calc.Function = 'sqrt(A*A + D*D + 2*B*B)'
brain = Show(calc)
brain.Representation = 'Surface'
ColorBy(brain, ('POINTS', 'Tensor Frobenius magnitude'))
brain.RescaleTransferFunctionToDataRange(True, False)
lut = GetColorTransferFunction('TensorFrobeniusmagnitude')
lut.ApplyPreset('Viridis (matplotlib)', True)
brain.Opacity = 0.94

wedges = Show(make_circles(wedge_pts, 'Wedges (radius 1)'))
wedges.DiffuseColor = [1.0, 1.0, 1.0]
wedges.AmbientColor = [1.0, 1.0, 1.0]
wedges.Ambient = 1.0
wedges.Diffuse = 0.0

trisectors = Show(make_circles(tri_pts, 'Trisectors (radius 1)'))
trisectors.DiffuseColor = [1.0, 0.34, 0.62]
trisectors.AmbientColor = [1.0, 0.34, 0.62]
trisectors.Ambient = 1.0
trisectors.Diffuse = 0.0

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 860]
view.InteractionMode = '2D'
view.OrientationAxesVisibility = 0
view.Background = [0.06, 0.07, 0.10]
view.BackgroundColorMode = 'Single Color'
view.CameraParallelProjection = 1
view.ResetCamera()
view.CameraParallelScale = 39
Render()
SaveScreenshot(output_file, view, ImageResolution=[1400,860])
SaveState('/workspace/brain_degenerate_points.pvsm')
