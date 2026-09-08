from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

input_file = '/workspace/QMCPACK.vti'
output_png = '/workspace/QMCPACK_persistence_0.04.png'

# Simplify the scalar field with an absolute persistence threshold of 0.04.
reader = XMLImageDataReader(FileName=[input_file])
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.UpdatePipeline()

# Four empirical quantiles split the simplified volume into five equal-volume bins.
data = servermanager.Fetch(simplified)
values = vtk_to_numpy(data.GetPointData().GetArray('Scalars_'))
levels = np.quantile(values, [0.2, 0.4, 0.6, 0.8]).tolist()

contours = Contour(Input=simplified)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = levels
contours.PointMergeMethod = 'Uniform Binning'

# Extract critical points of the simplified PL field.
critical = TTKScalarFieldCriticalPoints(Input=simplified)
critical.ScalarField = ['POINTS', 'Scalars_']
critical.UpdatePipeline()

view = CreateView('RenderView')
view.ViewSize = [1600, 1200]
view.Background = [0.055, 0.065, 0.085]
view.BackgroundColorMode = 'Gradient'
view.Background2 = [0.12, 0.14, 0.19]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 0

iso_display = Show(contours, view)
iso_display.Representation = 'Surface'
iso_display.DiffuseColor = [0.45, 0.72, 0.86]
iso_display.Opacity = 0.10
iso_display.Specular = 0.35
iso_display.SpecularPower = 30

# TTK CriticalType values: 0=min, 1=1-saddle, 2=2-saddle, 3=max, 4=degenerate.
categories = [
    (0, [0.10, 0.34, 1.00], 'minima'),
    (1, [1.00, 1.00, 1.00], '1-saddles'),
    (2, [1.00, 0.40, 0.03], '2-saddles'),
    (3, [0.95, 0.04, 0.05], 'maxima'),
    (4, [0.85, 0.30, 0.90], 'multi-saddles'),
]
for critical_type, color, name in categories:
    selected = Threshold(Input=critical)
    selected.Scalars = ['POINTS', 'CriticalType']
    selected.LowerThreshold = critical_type
    selected.UpperThreshold = critical_type
    selected.ThresholdMethod = 'Between'
    glyph = Glyph(Input=selected, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.35
    glyph.GlyphMode = 'All Points'
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.Specular = 0.55
    display.SpecularPower = 35
    display.Ambient = 0.25

outline = Outline(Input=simplified)
outline_display = Show(outline, view)
outline_display.DiffuseColor = [0.68, 0.72, 0.78]
outline_display.LineWidth = 1.5

view.ResetCamera()
camera = view.GetActiveCamera()
camera.Azimuth(32)
camera.Elevation(20)
camera.Zoom(1.25)
view.Update()
Render(view)
SaveScreenshot(output_png, view, ImageResolution=[1600, 1200], TransparentBackground=0)

# Save the state so the interactive ParaView scene is reproducible.
SaveState('/workspace/QMCPACK_persistence_0.04.pvsm')
print('quantile isovalues:', ', '.join(f'{v:.6f}' for v in levels))
print('wrote', output_png)
