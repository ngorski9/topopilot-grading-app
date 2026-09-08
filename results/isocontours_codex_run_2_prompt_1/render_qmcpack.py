from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# Read and simplify the scalar field with TTK's persistence-aware algorithm.
data = XMLImageDataReader(registrationName='QMCPACK', FileName=['/workspace/QMCPACK.vti'])
data.PointArrayStatus = ['Scalars_']
simplified = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.04)', Input=data)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.ThreadNumber = 8

# Piecewise-linear critical points of the simplified field.
critical = TTKScalarFieldCriticalPoints(registrationName='PL critical points', Input=simplified)
critical.ScalarField = ['POINTS', 'Scalars_']
critical.ThreadNumber = 8

# Use scalar-volume quantiles: each interval contains one fifth of the volume samples.
UpdatePipeline(proxy=simplified)
field = servermanager.Fetch(simplified).GetPointData().GetArray('Scalars_')
isovalues = np.quantile(vtk_to_numpy(field), [0.2, 0.4, 0.6, 0.8]).tolist()
contours = Contour(registrationName='4 equal-volume-range isocontours', Input=simplified)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = isovalues
contours.PointMergeMethod = 'Uniform Binning'

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [0.035, 0.045, 0.07]
view.BackgroundColorMode = 'Gradient'
view.Background2 = [0.12, 0.15, 0.21]
view.OrientationAxesVisibility = 0

# Isocontours are translucent so all critical points remain visible.
cv = Show(contours, view)
cv.Representation = 'Surface'
cv.ColorArrayName = [None, '']
cv.DiffuseColor = [0.42, 0.72, 0.82]
cv.Opacity = 0.18
cv.Specular = 0.35
cv.SpecularPower = 25

def show_type(code, color, name):
    t = Threshold(registrationName=name, Input=critical)
    t.Scalars = ['POINTS', 'CriticalType']
    t.ThresholdMethod = 'Between'
    t.LowerThreshold = code
    t.UpperThreshold = code
    g = Glyph(registrationName=name + ' glyphs', Input=t, GlyphType='Sphere')
    g.GlyphType.Radius = 1.0
    g.GlyphType.ThetaResolution = 18
    g.GlyphType.PhiResolution = 18
    g.ScaleFactor = 2.1
    g.GlyphMode = 'All Points'
    d = Show(g, view)
    d.DiffuseColor = color
    d.Specular = 0.7
    d.SpecularPower = 35
    return g

# TTK critical types: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
mins = show_type(0, [0.12, 0.32, 1.0], 'Minima')
s1 = show_type(1, [1.0, 1.0, 1.0], '1-saddles')
s2 = show_type(2, [1.0, 0.38, 0.05], '2-saddles')
maxs = show_type(3, [1.0, 0.04, 0.04], 'Maxima')
# Multi-saddles are retained as saddle points and use the 2-saddle color.
multi = show_type(4, [1.0, 0.38, 0.05], 'Multi-saddles')

title = Text(registrationName='Title')
title.Text = 'QMCPACK: persistence simplification threshold = 0.04'
tv = Show(title, view)
tv.WindowLocation = 'Upper Center'
tv.FontSize = 20
tv.Color = [0.96, 0.96, 0.96]

legend = Text(registrationName='Legend')
legend.Text = ('Critical points:  red = maxima    orange = 2-saddles    white = 1-saddles    blue = minima\n'
               'Equal-volume isocontours: ' + ', '.join(f'{v:.5f}' for v in isovalues))
lv = Show(legend, view)
lv.WindowLocation = 'Lower Left Corner'
lv.FontSize = 15
lv.Color = [0.95, 0.95, 0.95]

view.CameraPosition = [205, -250, 205]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0.2, 0.35, 0.92]
view.CameraParallelScale = 135
view.CameraParallelProjection = 1
Render(view)
SaveScreenshot('/workspace/output/QMCPACK_persistence_004.png', view, ImageResolution=[1400, 1000])
SaveState('/workspace/output/QMCPACK_persistence_004.pvsm')
print('Wrote /workspace/output/QMCPACK_persistence_004.png')
