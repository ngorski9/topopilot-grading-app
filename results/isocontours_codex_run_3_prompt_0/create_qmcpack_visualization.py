from paraview.simple import *
from paraview import servermanager
import numpy as np

# Read and simplify the scalar field with the requested absolute persistence.
source = XMLImageDataReader(registrationName='QMCPACK.vti', FileName=['/workspace/QMCPACK.vti'])
source.PointArrayStatus = ['Scalars_']

simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.04)', Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.PairType = 0

# The four isovalues are the 20/40/60/80 percent quantiles of the simplified
# field: this divides the sampled volume into five equal-size scalar regions.
UpdatePipeline(proxy=simplified)
simp_data = servermanager.Fetch(simplified)
values = np.array([simp_data.GetPointData().GetArray('Scalars_').GetValue(i)
                   for i in range(simp_data.GetNumberOfPoints())])
iso_values = [float(x) for x in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]

contours = Contour(registrationName='4 equal-volume isocontours', Input=simplified)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = iso_values
contours.PointMergeMethod = 'Uniform Binning'

critical = TTKScalarFieldCriticalPoints(
    registrationName='All PL critical points', Input=simplified)
critical.ScalarField = ['POINTS', 'Scalars_']
critical.InputOffsetField = ['POINTS', 'Scalars__Order']
critical.ForceInputOffsetField = 1

view = CreateView('RenderView')
view.ViewSize = [1500, 1000]
view.Background = [0.055, 0.055, 0.07]
view.BackgroundColorMode = 'Gradient'
view.Background2 = [0.16, 0.16, 0.20]

# Show the simplified volume only as a quiet, translucent contextual shell.
outline = Outline(registrationName='Volume extent', Input=simplified)
outline_display = Show(outline, view)
outline_display.DiffuseColor = [0.45, 0.45, 0.48]
outline_display.LineWidth = 1.5

contour_display = Show(contours, view)
contour_display.Representation = 'Surface'
contour_display.DiffuseColor = [0.20, 0.80, 0.82]
contour_display.Opacity = 0.48
contour_display.LineWidth = 2.5

# CriticalType: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
# Separate selections give a clear, fixed categorical palette.
categories = [
    ('Minima (blue)', 0, [0.10, 0.35, 1.00]),
    ('1-saddles (white)', 1, [1.00, 1.00, 1.00]),
    ('2-saddles (orange)', 2, [1.00, 0.42, 0.05]),
    ('Maxima (red)', 3, [1.00, 0.06, 0.05]),
]
for name, critical_type, color in categories:
    selected = Threshold(registrationName=name, Input=critical)
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.LowerThreshold = critical_type
    selected.UpperThreshold = critical_type
    selected.ThresholdMethod = 'Between'
    display = Show(selected, view)
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = 1.5
    display.Opacity = 1.0
    display.DiffuseColor = color
    display.AmbientColor = color
    display.ColorArrayName = [None, '']

# Multi-saddles are still PL critical points; display them with the 1-saddle
# convention so none of the computed critical points is omitted.
multi = Threshold(registrationName='Multi-saddles (white)', Input=critical)
multi.Scalars = ['POINTS', 'CriticalType']
multi.LowerThreshold = 4
multi.UpperThreshold = 255
multi.ThresholdMethod = 'Between'
multi_display = Show(multi, view)
multi_display.Representation = 'Point Gaussian'
multi_display.GaussianRadius = 1.5
multi_display.DiffuseColor = [1.0, 1.0, 1.0]
multi_display.AmbientColor = [1.0, 1.0, 1.0]
multi_display.ColorArrayName = [None, '']

view.OrientationAxesVisibility = 1
view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle = 'X'
view.AxesGrid.YTitle = 'Y'
view.AxesGrid.ZTitle = 'Z'
view.AxesGrid.GridColor = [0.35, 0.35, 0.35]
view.CameraPosition = [207, -250, 250]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = 115

Render(view)
SaveScreenshot('/workspace/QMCPACK_topology.png', view, ImageResolution=[1500, 1000])
SaveState('/workspace/QMCPACK_topology.pvsm')

with open('/workspace/QMCPACK_topology_summary.txt', 'w') as f:
    f.write('Persistence threshold (absolute): 0.04\\n')
    f.write('Equal-volume isovalues: ' + ', '.join('%.9g' % x for x in iso_values) + '\\n')
    f.write('Critical point colors: maxima=red, 2-saddles=orange, 1-saddles=white, minima=blue\\n')
