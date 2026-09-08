from paraview.simple import *
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk
import numpy as np

input_file = '/workspace/brain.vti'
points_file = '/workspace/brain_degenerate_points.vtp'
image_file = '/workspace/brain_degenerate_points.png'
state_file = '/workspace/brain_degenerate_points.pvsm'

# Read the original tensor components and locate the zeros of the anisotropy
# vector (A-D, 2B) in each triangular half-pixel.  Its orientation distinguishes
# the two generic tensor singularities: positive = wedge, negative = trisector.
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
im = reader.GetOutput()
pd = im.GetPointData()
A = vtk_to_numpy(pd.GetArray('A'))
B = vtk_to_numpy(pd.GetArray('B'))
D = vtk_to_numpy(pd.GetArray('D'))
extent = im.GetExtent()
nx, ny = extent[1] - extent[0] + 1, extent[3] - extent[2] + 1
origin, spacing = im.GetOrigin(), im.GetSpacing()

F = np.c_[A-D, 2.0*B].reshape(ny, nx, 2)
outpts, kinds = [], []
eps = 1e-10
for j in range(ny - 1):
    for i in range(nx - 1):
        corners = [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]
        for tri in ((0,1,2), (0,2,3)):
            q = np.array([corners[k] for k in tri], dtype=float)
            v = np.array([F[int(y), int(x)] for x,y in q])
            M = np.column_stack((v[1]-v[0], v[2]-v[0]))
            det = np.linalg.det(M)
            if abs(det) < eps:
                continue
            uv = np.linalg.solve(M, -v[0])
            bary = np.array([1.0-uv.sum(), uv[0], uv[1]])
            # Half-open test suppresses duplicate roots on shared cell edges.
            if np.all(bary >= -eps) and np.all(bary <= 1+eps):
                p = bary @ q
                if not any(np.linalg.norm(p-r[0]) < 1e-6 for r in outpts):
                    outpts.append((p, det))

poly = vtk.vtkPolyData()
vtkpts = vtk.vtkPoints()
typ = vtk.vtkIntArray(); typ.SetName('Type')
verts = vtk.vtkCellArray()
for p, det in outpts:
    point_id = vtkpts.InsertNextPoint(origin[0]+p[0]*spacing[0], origin[1]+p[1]*spacing[1], 0.5)
    verts.InsertNextCell(1); verts.InsertCellPoint(point_id)
    typ.InsertNextValue(1 if det > 0 else 0) # 1 wedge, 0 trisector
poly.SetPoints(vtkpts); poly.SetVerts(verts); poly.GetPointData().AddArray(typ)
writer = vtk.vtkXMLPolyDataWriter(); writer.SetFileName(points_file); writer.SetInputData(poly); writer.Write()
print('Degenerate points:', len(outpts), 'wedges:', sum(d>0 for _,d in outpts), 'trisectors:', sum(d<0 for _,d in outpts))

# Render the original field as an A-component surface, then overlay spherical
# degenerate-point markers with the requested radius and colors.
field = XMLImageDataReader(registrationName='Brain tensor field', FileName=[input_file])
view = CreateView('RenderView')
view.ViewSize = [1400, 850]
view.Background = [0.04, 0.05, 0.08]
field_display = Show(field, view)
ColorBy(field_display, ('POINTS', 'A'))
field_display.Representation = 'Surface'
field_display.Opacity = 0.82
field_display.SetScalarBarVisibility(view, True)
LUT = GetColorTransferFunction('A'); LUT.ApplyPreset('Cool to Warm', True)

deg = XMLPolyDataReader(registrationName='Degenerate points', FileName=[points_file])
tris = Threshold(registrationName='Trisectors', Input=deg)
tris.Scalars = ['POINTS', 'Type']; tris.LowerThreshold = -0.5; tris.UpperThreshold = 0.5
wedges = Threshold(registrationName='Wedges', Input=deg)
wedges.Scalars = ['POINTS', 'Type']; wedges.LowerThreshold = 0.5; wedges.UpperThreshold = 1.5
for source, name, color in ((tris, 'Trisectors (pink)', [1.0, 0.20, 0.65]), (wedges, 'Wedges (white)', [1.0, 1.0, 1.0])):
    sphere = Glyph(registrationName=name, Input=source, GlyphType='Sphere')
    sphere.GlyphType.Radius = 1.0
    sphere.ScaleArray = ['POINTS', 'No scale array']
    sphere.ScaleFactor = 1.0
    display = Show(sphere, view)
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.25
    display.ColorArrayName = [None, '']

view.CameraParallelProjection = 1
view.CameraPosition = [53.5, 32.5, 180]
view.CameraFocalPoint = [53.5, 32.5, 0]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = 72
Render()
SaveScreenshot(image_file, view, ImageResolution=[1400, 850])
SaveState(state_file)
