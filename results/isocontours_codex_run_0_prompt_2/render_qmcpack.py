from paraview.simple import *
from paraview import servermanager
from paraview.vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import csv

src = XMLImageDataReader(registrationName='QMCPACK.vti', FileName=['/workspace/QMCPACK.vti'])
simp = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.04)', Input=src)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 1
simp.UpdatePipeline()

# Equal-volume bands: quantiles of the simplified scalar field.
simp_data = servermanager.Fetch(simp)
values = vtk_to_numpy(simp_data.GetPointData().GetArray('Scalars_'))
levels = np.quantile(values, [0.2, 0.4, 0.6, 0.8]).tolist()

contours = Contour(registrationName='Four equal-volume isocontours', Input=simp)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = levels
contours.PointMergeMethod = 'Uniform Binning'

critical = TTKScalarFieldCriticalPoints(registrationName='All PL critical points', Input=simp)
critical.ScalarField = ['POINTS', 'Scalars_']
critical.UpdatePipeline()
critical_data = servermanager.Fetch(critical)

# Persist the complete piecewise-linear critical-point table.
ctype = vtk_to_numpy(critical_data.GetPointData().GetArray('CriticalType'))
scalar = vtk_to_numpy(critical_data.GetPointData().GetArray('Scalars_'))
points = vtk_to_numpy(critical_data.GetPoints().GetData())
labels = {0: 'Minimum', 1: '1-saddle', 2: '2-saddle', 3: 'Maximum'}
with open('/workspace/QMCPACK_critical_points_threshold_0.04.csv', 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['id', 'x', 'y', 'z', 'scalar', 'critical_type', 'type_name'])
    for i, (p, v, t) in enumerate(zip(points, scalar, ctype)):
        out.writerow([i, *p, v, int(t), labels.get(int(t), 'Multi-saddle')])

view = CreateView('RenderView')
view.ViewSize = [1800, 1200]
view.Background = [0.055, 0.065, 0.09]
view.OrientationAxesVisibility = 0

contour_display = Show(contours, view)
contour_display.Representation = 'Surface'
ColorBy(contour_display, None)
contour_display.DiffuseColor = [0.25, 0.72, 0.82]
contour_display.Opacity = 0.34
contour_display.LineWidth = 2.5

# Spheres make every PL critical point visibly distinct.
glyphs = Glyph(registrationName='Critical point spheres', Input=critical, GlyphType='Sphere')
glyphs.ScaleArray = ['POINTS', 'No scale array']
glyphs.ScaleFactor = 0.65
glyphs.GlyphTransform = 'Transform2'
glyph_display = Show(glyphs, view)
ColorBy(glyph_display, ('POINTS', 'CriticalType'))
glyph_display.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction('CriticalType')
lut.InterpretValuesAsCategories = 1
lut.Annotations = ['0', 'Minima', '1', '1-saddles', '2', '2-saddles', '3', 'Maxima', '4', 'Multi-saddles']
lut.IndexedColors = [0.10, 0.32, 0.95, 1.0, 1.0, 1.0, 1.0, 0.42, 0.05, 0.92, 0.05, 0.05, 1.0, 0.42, 0.05]
bar = GetScalarBar(lut, view)
bar.Title = 'PL critical points'
bar.ComponentTitle = ''
bar.DrawAnnotations = 1

view.ResetCamera()
view.CameraPosition = [72, -86, 68]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 0
Render(view)
SaveScreenshot('/workspace/QMCPACK_simplified_critical_points.png', view, ImageResolution=[1800, 1200])
SaveState('/workspace/QMCPACK_simplified_visualization.pvsm')

counts = {labels.get(int(t), 'Multi-saddle'): int((ctype == t).sum()) for t in np.unique(ctype)}
with open('/workspace/QMCPACK_visualization_summary.txt', 'w') as f:
    f.write('Persistence threshold: 0.04 (absolute)\n')
    f.write('Equal-volume isovalues: ' + ', '.join(f'{x:.9g}' for x in levels) + '\n')
    for name, count in counts.items(): f.write(f'{name}: {count}\n')
print('levels', levels)
print('counts', counts)
