from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

# Load TTK before creating its filters.
LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so',
           remote=False, ns=globals())

view = CreateView('RenderView')
view.ViewSize = [1600, 1000]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0

volume = XMLImageDataReader(registrationName='QMCPACK.vti',
                            FileName=['/workspace/QMCPACK.vti'])

# Persistence simplification: absolute persistence threshold = 0.04.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.04)', Input=volume)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1

# Quantiles of the simplified volume produce five equal-size value regions.
simplified.UpdatePipeline()
dataset = servermanager.Fetch(simplified)
values = vtk_to_numpy(dataset.GetPointData().GetArray('Scalars_'))
levels = [float(x) for x in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]

contours = Contour(registrationName='Four equal-volume isocontours', Input=simplified)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = levels
contours.ComputeScalars = 1
contour_display = Show(contours, view)
contour_display.Representation = 'Surface'
contour_display.DiffuseColor = [0.70, 0.72, 0.76]
contour_display.Opacity = 0.30

# All critical points of the simplified PL scalar field.
msc = TTKMorseSmaleComplex(registrationName='All PL critical points', Input=simplified)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.ThresholdIsAbsolute = 1
msc.SaddleConnectorsPersistenceThreshold = 0.0

# Port 0 of TTK Morse-Smale Complex is the critical-point output.
critical_points = OutputPort(msc, 0)
cp_display = Show(critical_points, view)
ColorBy(cp_display, ('POINTS', 'CellDimension'))
cp_display.Representation = 'Points'
cp_display.RenderPointsAsSpheres = 1
cp_display.PointSize = 12

# Dimension encodes PL critical-point type: 0 minimum, 1 1-saddle,
# 2 2-saddle, 3 maximum.
cp_lut = GetColorTransferFunction('CellDimension')
cp_lut.InterpretValuesAsCategories = 1
cp_lut.Annotations = ['0', 'Minimum', '1', '1-saddle',
                      '2', '2-saddle', '3', 'Maximum']
cp_lut.IndexedColors = [0.10, 0.35, 1.00,   # minima: blue
                         1.00, 1.00, 1.00,   # 1-saddles: white
                         1.00, 0.50, 0.00,   # 2-saddles: orange
                         1.00, 0.05, 0.05]   # maxima: red
cp_lut.IndexedOpacities = [1.0, 1.0, 1.0, 1.0]
cp_display.LookupTable = cp_lut
cp_display.SetScalarBarVisibility(view, True)
bar = GetScalarBar(cp_lut, view)
bar.Title = 'PL critical point'
bar.ComponentTitle = ''
bar.DrawAnnotations = 1
bar.WindowLocation = 'Upper Right Corner'

view.CameraParallelProjection = 0
view.ResetCamera()
Render(view)

SaveScreenshot('/workspace/QMCPACK_topology.png', view, ImageResolution=[1600, 1000])
SaveState('/workspace/QMCPACK_topology.pvsm')
print('Equal-volume isocontour levels:', levels)
