from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

src = XMLImageDataReader(registrationName='QMCPACK', FileName=['/workspace/QMCPACK.vti'])
src.PointArrayStatus = ['Scalars_']

# Cancel all persistence pairs below the requested absolute threshold.
simp = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.04)', Input=src)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 1
simp.PairType = 0
simp.UseAllCores = 1

UpdatePipeline(proxy=simp)
out = servermanager.Fetch(simp)
arr = out.GetPointData().GetArray('Scalars_')
if arr is None:
    # TTK retains the selected field; this fallback finds its scalar output.
    arr = out.GetPointData().GetScalars()
vals = vtk_to_numpy(arr)
levels = np.quantile(vals, [0.2, 0.4, 0.6, 0.8]).tolist()
print('SIMPLIFIED_RANGE', float(vals.min()), float(vals.max()))
print('EQUAL_VOLUME_ISOVALUES', levels)

cp = TTKScalarFieldCriticalPoints(registrationName='Piecewise-linear critical points', Input=simp)
cp.ScalarField = ['POINTS', arr.GetName()]
cp.UseAllCores = 1
UpdatePipeline(proxy=cp)
cpout = servermanager.Fetch(cp)
print('CP_POINT_ARRAYS', [cpout.GetPointData().GetArrayName(i) for i in range(cpout.GetPointData().GetNumberOfArrays())])
print('CP_CELL_ARRAYS', [cpout.GetCellData().GetArrayName(i) for i in range(cpout.GetCellData().GetNumberOfArrays())])

cont = Contour(registrationName='Four equal-volume isocontours', Input=simp)
cont.ContourBy = ['POINTS', arr.GetName()]
cont.Isosurfaces = levels
cont.PointMergeMethod = 'Uniform Binning'

view = CreateView('RenderView')
view.ViewSize = [1600, 1100]
view.Background = [0.055, 0.065, 0.09]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0

cd = Show(cont, view)
cd.Representation = 'Surface'
cd.DiffuseColor = [0.35, 0.72, 0.82]
cd.Opacity = 0.30
cd.Specular = 0.35
cd.SpecularPower = 25

# TTK critical type convention: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
colors = {0: (0.10, 0.35, 1.0), 1: (1.0, 1.0, 1.0), 2: (1.0, 0.42, 0.05), 3: (1.0, 0.05, 0.05), 4: (1.0, 0.86, 0.05)}
names = {0: 'Minima', 1: '1-saddles', 2: '2-saddles', 3: 'Maxima', 4: 'Multi-saddles'}
# Type 4 is TTK's degenerate multi-saddle type; draw it separately so every
# reported piecewise-linear critical point is represented.
for typ in (0, 1, 2, 3, 4):
    th = Threshold(registrationName=names[typ], Input=cp)
    th.Scalars = ['POINTS', 'CriticalType']
    th.ThresholdMethod = 'Between'
    th.LowerThreshold = typ
    th.UpperThreshold = typ
    glyph = Glyph(registrationName=names[typ] + ' glyphs', Input=th, GlyphType='Sphere')
    glyph.GlyphType.Radius = 1.65
    glyph.GlyphType.ThetaResolution = 20
    glyph.GlyphType.PhiResolution = 20
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    d = Show(glyph, view)
    d.DiffuseColor = colors[typ]
    d.AmbientColor = colors[typ]
    d.Ambient = 0.28
    d.Specular = 0.7
    d.SpecularPower = 32

view.CameraPosition = [175, -205, 165]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 0
view.CameraViewAngle = 28
view.ResetCamera()
Render(view)
SaveScreenshot('/workspace/QMCPACK_persistence_0.04.png', view, ImageResolution=[1600,1100])
SaveData('/workspace/QMCPACK_simplified_0.04.vti', proxy=simp)
SaveData('/workspace/QMCPACK_critical_points_0.04.vtp', proxy=cp)
SaveState('/workspace/QMCPACK_persistence_0.04.pvsm')
