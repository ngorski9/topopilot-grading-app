from paraview.simple import *
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np

src = XMLImageDataReader(registrationName='QMCPACK', FileName=['/workspace/QMCPACK.vti'])
src.PointArrayStatus = ['Scalars_']
UpdatePipeline(proxy=src)

# Persistence-based topological simplification requested by the user.
simp = TTKTopologicalSimplificationByPersistence(registrationName='Persistence threshold 0.04', Input=src)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 1
UpdatePipeline(proxy=simp)

# Inspect the output so subsequent filters use the exact retained scalar name.
simp_data = servermanager.Fetch(simp)
names = [simp_data.GetPointData().GetArrayName(i) for i in range(simp_data.GetPointData().GetNumberOfArrays())]
scalar_name = 'Scalars_' if 'Scalars_' in names else names[0]
values = vtk_to_numpy(simp_data.GetPointData().GetArray(scalar_name))
levels = [float(x) for x in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]
print('Simplified scalar array:', scalar_name)
print('Equal-volume isovalues:', levels)

crit = TTKScalarFieldCriticalPoints(registrationName='Piecewise-linear critical points', Input=simp)
crit.ScalarField = ['POINTS', scalar_name]
crit.ForceInputOffsetField = 0
crit.Withvertexidentifiers = 1
crit.Withvertexscalars = 1
UpdatePipeline(proxy=crit)
crit_data = servermanager.Fetch(crit)
pd = crit_data.GetPointData()
print('Critical point arrays:', [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())])

# Save machine-readable results and the simplified volume.
SaveData('/workspace/QMCPACK_simplified_0.04.vti', proxy=simp)
SaveData('/workspace/QMCPACK_critical_points_0.04.vtp', proxy=crit)

cont = Contour(registrationName='4 equal-volume isocontours', Input=simp)
cont.ContourBy = ['POINTS', scalar_name]
cont.Isosurfaces = levels
cont.PointMergeMethod = 'Uniform Binning'

view = CreateRenderView()
view.ViewSize = [1600, 1050]
view.Background = [0.06, 0.07, 0.10]
view.UseColorPaletteForBackground = 0

cont_disp = Show(cont, view)
cont_disp.Representation = 'Surface'
cont_disp.DiffuseColor = [0.35, 0.72, 0.92]
cont_disp.Opacity = 0.32
cont_disp.LineWidth = 1.5

# TTK encodes critical-point Morse index in the 'CriticalType' field:
# 0=min, 1=1-saddle, 2=2-saddle, 3=max.  Separate them for the requested colors.
colors = {0: (0.15, 0.35, 1.0), 1: (1.0, 1.0, 1.0), 2: (1.0, 0.45, 0.05), 3: (1.0, 0.05, 0.05), 4: (0.55, 0.55, 0.55)}
labels = {0: 'Minima', 1: '1-saddles', 2: '2-saddles', 3: 'Maxima', 4: 'Degenerate critical points'}
for ctype in (0, 1, 2, 3, 4):
    th = Threshold(registrationName=labels[ctype], Input=crit)
    th.Scalars = ['POINTS', 'CriticalType']
    th.LowerThreshold = ctype
    th.UpperThreshold = ctype
    th.ThresholdMethod = 'Between'
    glyph = Glyph(registrationName=labels[ctype] + ' glyphs', Input=th, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 2.3
    glyph.GlyphMode = 'All Points'
    d = Show(glyph, view)
    d.DiffuseColor = colors[ctype]
    d.AmbientColor = colors[ctype]
    d.Ambient = 0.25
    d.Specular = 0.35

outline = Outline(Input=simp)
od = Show(outline, view)
od.DiffuseColor = [0.7, 0.7, 0.7]

view.CameraPosition = [190, -235, 185]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = 95
Render(view)
SaveScreenshot('/workspace/QMCPACK_persistence_0.04_critical_points.png', view, ImageResolution=[1600, 1050])
SaveState('/workspace/QMCPACK_persistence_0.04_visualization.pvsm')

# Report each output critical point (type, coordinates, scalar) for direct inspection.
ctype_arr = pd.GetArray('CriticalType')
scalar_arr = pd.GetArray('CellDimension') or pd.GetArray('VertexScalar') or pd.GetArray(scalar_name)
for i in range(crit_data.GetNumberOfPoints()):
    p = crit_data.GetPoint(i)
    t = int(ctype_arr.GetTuple1(i)) if ctype_arr else -1
    val = scalar_arr.GetTuple1(i) if scalar_arr else float('nan')
    print('%s: x=%.6g y=%.6g z=%.6g scalar=%.9g' % (labels.get(t, 'Unknown'), p[0], p[1], p[2], val))
