"""Create a ParaView-ready eigenvector partition visualization for Ocean.vti."""
import math
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from paraview.simple import *
from paraview import servermanager

INFILE = '/workspace/Ocean.vti'
PARTITION_FILE = '/workspace/Ocean_eigenvector_partition.vti'
POINTS_FILE = '/workspace/Ocean_degenerate_points.vtp'
PNG_FILE = '/workspace/Ocean_eigenvector_partition.png'
STATE_FILE = '/workspace/Ocean_eigenvector_partition.pvsm'

# Read the original piecewise-linear tensor field.
reader = XMLImageDataReader(registrationName='Ocean tensor field', FileName=[INFILE])
reader.UpdatePipeline()
data = servermanager.Fetch(reader)
dims = data.GetDimensions()
nx, ny = dims[0], dims[1]
pd = data.GetPointData()
A = vtk_to_numpy(pd.GetArray('A')).reshape((ny, nx))
B = vtk_to_numpy(pd.GetArray('B')).reshape((ny, nx))
C = vtk_to_numpy(pd.GetArray('C')).reshape((ny, nx))
D = vtk_to_numpy(pd.GetArray('D')).reshape((ny, nx))

# The unoriented principal eigenvector is governed by the symmetric part.
# q,r are the standard double-angle coordinates of its line field.
q = A - D
r = B + C
angle = 0.5 * np.arctan2(r, q)  # line angle, modulo pi

# 10 samples per 1x1 input cell: 1000 x 1000 cells / 1001 x 1001 vertices.
scale = 10
out_nx, out_ny = (nx - 1) * scale + 1, (ny - 1) * scale + 1
xx = np.arange(out_nx) / scale
yy = np.arange(out_ny) / scale
gx, gy = np.meshgrid(xx, yy)
i = np.minimum(gx.astype(int), nx - 2)
j = np.minimum(gy.astype(int), ny - 2)
u, v = gx - i, gy - j
def bilinear(field):
    return ((1-u)*(1-v)*field[j,i] + u*(1-v)*field[j,i+1]
            + (1-u)*v*field[j+1,i] + u*v*field[j+1,i+1])
Q, R = bilinear(q), bilinear(r)
partition = (0.5 * np.arctan2(R, Q) % math.pi) / math.pi

# Store a scalar partition; HSV-like cyclic color map will be attached in ParaView.
out = vtk.vtkImageData()
out.SetOrigin(0, 0, 0)
out.SetSpacing(1.0 / scale, 1.0 / scale, 1)
out.SetDimensions(out_nx, out_ny, 1)
arr = numpy_to_vtk(np.ascontiguousarray(partition.ravel()), deep=True)
arr.SetName('Eigenvector partition (line angle)')
out.GetPointData().SetScalars(arr)
w = vtk.vtkXMLImageDataWriter(); w.SetFileName(PARTITION_FILE); w.SetInputData(out); w.Write()

# Locate PL degeneracies.  Each cell is split into two triangles; within a
# triangle q and r are affine, so their common zero is exact.  Index sign
# determines the conventional wedge (+1/2) vs trisector (-1/2) type.
pts, kinds = [], []
triangles = [((0,0),(1,0),(1,1)), ((0,0),(1,1),(0,1))]
for y in range(ny-1):
    for x in range(nx-1):
        for tri in triangles:
            xy = np.asarray(tri, dtype=float)
            fv = np.array([q[y+dy,x+dx] for dx,dy in tri])
            gv = np.array([r[y+dy,x+dx] for dx,dy in tri])
            M = np.array([[fv[1]-fv[0], fv[2]-fv[0]], [gv[1]-gv[0], gv[2]-gv[0]]])
            rhs = -np.array([fv[0], gv[0]])
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue
            uv = np.linalg.solve(M, rhs)
            bary = np.array([1-uv[0]-uv[1], uv[0], uv[1]])
            if np.min(bary) >= -1e-9 and np.max(bary) <= 1+1e-9:
                loc = bary @ xy
                pts.append((x+loc[0], y+loc[1], 0.15))
                kinds.append(0 if det > 0 else 1)  # wedge, trisector

poly = vtk.vtkPolyData(); vp = vtk.vtkPoints()
for p in pts: vp.InsertNextPoint(*p)
poly.SetPoints(vp)
verts = vtk.vtkCellArray()
for k in range(len(pts)):
    verts.InsertNextCell(1); verts.InsertCellPoint(k)
poly.SetVerts(verts)
ka = vtk.vtkIntArray(); ka.SetName('Degenerate type')
for k in kinds: ka.InsertNextValue(k)
poly.GetPointData().AddArray(ka); poly.GetPointData().SetScalars(ka)
pw = vtk.vtkXMLPolyDataWriter(); pw.SetFileName(POINTS_FILE); pw.SetInputData(poly); pw.Write()

# Build final ParaView scene.
Hide(reader)
part = XMLImageDataReader(registrationName='Eigenvector partition — 10 px per square', FileName=[PARTITION_FILE])
part_display = Show(part)
part_display.Representation = 'Surface'
ColorBy(part_display, ('POINTS', 'Eigenvector partition (line angle)'))
lut = GetColorTransferFunction('Eigenvectorpartitionlineangle')
lut.RGBPoints = [0.0, 0.20,0.08,0.45, 0.25,0.05,0.55,0.75, 0.5,0.15,0.78,0.52, 0.75,0.95,0.55,0.12, 1.0,0.20,0.08,0.45]
lut.ColorSpace = 'HSV'
lut.RescaleTransferFunction(0, 1)
part_display.LookupTable = lut

deg = XMLPolyDataReader(registrationName='Degenerate points — radius 1', FileName=[POINTS_FILE])
glyph = Glyph(registrationName='Degenerate points (radius 1)', Input=deg, GlyphType='Sphere')
glyph.GlyphType.Radius = 1.0
glyph.GlyphType.ThetaResolution = 24
glyph.GlyphType.PhiResolution = 16
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph_display = Show(glyph)
ColorBy(glyph_display, ('POINTS', 'Degenerate type'))
dlut = GetColorTransferFunction('Degeneratetype')
dlut.InterpretValuesAsCategories = 1
dlut.Annotations = ['0', 'Wedge', '1', 'Trisector']
dlut.IndexedColors = [1.0,1.0,1.0, 1.0,0.41,0.71]
glyph_display.LookupTable = dlut
glyph_display.SetScalarBarVisibility(GetActiveViewOrCreate('RenderView'), True)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1000, 1000]
view.InteractionMode = '2D'
view.CameraParallelProjection = 1
view.CameraPosition = [50, 50, 200]
view.CameraFocalPoint = [50, 50, 0]
view.CameraParallelScale = 52
view.Background = [0.08, 0.08, 0.10]
view.UseColorPaletteForBackground = 0
Render()
SaveScreenshot(PNG_FILE, view, ImageResolution=[1000,1000])
SaveState(STATE_FILE)
print('Degenerate points:', len(pts), 'wedges:', kinds.count(0), 'trisectors:', kinds.count(1))
print(PNG_FILE)
