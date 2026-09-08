import vtk
from paraview.simple import *

input_file = '/workspace/brain.vti'
points_file = '/workspace/brain_degenerate_points.vtp'
state_file = '/workspace/brain_degenerate_visualization.pvsm'

# Extract roots of the linearly interpolated traceless-tensor components in
# the two triangles of each image-grid pixel.
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
image = reader.GetOutput()
nx, ny, nz = image.GetDimensions()
pd = image.GetPointData()
a, b, d = (pd.GetArray(name) for name in ('A', 'B', 'D'))

out_points = vtk.vtkPoints()
types = vtk.vtkStringArray(); types.SetName('SingularityType')
indices = vtk.vtkDoubleArray(); indices.SetName('Index')

def add_triangle(ids):
    # f=(A-D, 2B); solve f0 + u(f1-f0) + v(f2-f0)=0.
    vals = [(a.GetTuple1(i)-d.GetTuple1(i), 2*b.GetTuple1(i)) for i in ids]
    x0, y0 = vals[0]
    m00, m01 = vals[1][0]-x0, vals[2][0]-x0
    m10, m11 = vals[1][1]-y0, vals[2][1]-y0
    det = m00*m11-m01*m10
    if abs(det) < 1.e-18:
        return
    u = (-x0*m11 + m01*y0)/det
    v = (-m00*y0 + x0*m10)/det
    eps = 1.e-8
    if u < -eps or v < -eps or u+v > 1+eps:
        return
    p0, p1, p2 = (image.GetPoint(i) for i in ids)
    xyz = [p0[k] + u*(p1[k]-p0[k]) + v*(p2[k]-p0[k]) for k in range(3)]
    # The sign of the local map is the line-field index times two.
    # Positive is a wedge (+1/2); negative is a trisector (-1/2).
    index = 0.5 if det > 0 else -0.5
    out_points.InsertNextPoint(xyz)
    types.InsertNextValue('Wedge' if det > 0 else 'Trisector')
    indices.InsertNextValue(index)

for j in range(ny-1):
    for i in range(nx-1):
        p00 = j*nx+i; p10 = p00+1; p01 = p00+nx; p11 = p01+1
        add_triangle((p00, p10, p11))
        add_triangle((p00, p11, p01))

poly = vtk.vtkPolyData(); poly.SetPoints(out_points)
verts = vtk.vtkCellArray()
for i in range(out_points.GetNumberOfPoints()):
    verts.InsertNextCell(1); verts.InsertCellPoint(i)
poly.SetVerts(verts)
poly.GetPointData().AddArray(types); poly.GetPointData().AddArray(indices)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(points_file); writer.SetInputData(poly); writer.Write()
print('degenerate points:', out_points.GetNumberOfPoints())
print('wedges:', sum(types.GetValue(i)=='Wedge' for i in range(types.GetNumberOfValues())))
print('trisectors:', sum(types.GetValue(i)=='Trisector' for i in range(types.GetNumberOfValues())))

# Build a clean ParaView scene with separated point sets for fixed category colors.
data = XMLImageDataReader(registrationName='brain tensor field', FileName=[input_file])
data_display = Show(data)
data_display.Representation = 'Surface'
data_display.ColorArrayName = ['POINTS', 'A']
ColorBy(data_display, ('POINTS', 'A'))
data_display.Opacity = 0.72
data_display.RescaleTransferFunctionToDataRange(True, False)

deg = XMLPolyDataReader(registrationName='degenerate points', FileName=[points_file])
wedges = Threshold(registrationName='wedges', Input=deg)
wedges.Scalars = ['POINTS', 'Index']; wedges.LowerThreshold = 0.49; wedges.UpperThreshold = 0.51; wedges.ThresholdMethod = 'Between'
trisectors = Threshold(registrationName='trisectors', Input=deg)
trisectors.Scalars = ['POINTS', 'Index']; trisectors.LowerThreshold = -0.51; trisectors.UpperThreshold = -0.49; trisectors.ThresholdMethod = 'Between'

for src, name, color in ((wedges, 'Wedges (radius 1)', [1.0,1.0,1.0]), (trisectors, 'Trisectors (radius 1)', [1.0,0.35,0.65])):
    glyph = Glyph(registrationName=name, Input=src, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']; glyph.ScaleFactor = 1.0
    glyph.GlyphType.Radius = 1.0
    disp = Show(glyph); disp.DiffuseColor = color; disp.AmbientColor = color; disp.Ambient = 0.35

Hide(deg); Hide(wedges); Hide(trisectors)
view = GetActiveViewOrCreate('RenderView')
view.InteractionMode = '2D'
view.Background = [0.08, 0.08, 0.08]
view.UseColorPaletteForBackground = 0
view.CameraParallelProjection = 1
view.ViewSize = [1100, 700]
ResetCamera(view)
Render(view)
SaveState(state_file)
