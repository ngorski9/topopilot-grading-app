from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import csv

paraview.simple._DisableFirstRenderCameraReset()
data = XMLImageDataReader(registrationName='QMCPACK.vti', FileName=['/workspace/QMCPACK.vti'])
data.PointArrayStatus = ['Scalars_']
data.UpdatePipeline()

# Persistence-driven simplification of the input scalar field.
simp = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.04)', Input=data)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 1
simp.UpdatePipeline()

# Extract PL critical points from the simplified field.
msc = TTKMorseSmaleComplex(registrationName='PL critical points', Input=simp)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.ThresholdIsAbsolute = 1
msc.SaddleConnectorsPersistenceThreshold = 0.04
msc.CriticalPoints = 1
msc.AscendingSegmentation = 0
msc.DescendingSegmentation = 0
msc.Ascending1Separatrices = 0
msc.Descending1Separatrices = 0
msc.Ascending2Separatrices = 0
msc.Descending2Separatrices = 0
msc.UpdatePipeline()

# Four quantile isovalues give five equal-count scalar regions in the volume.
arr = vtk_to_numpy(servermanager.Fetch(simp).GetPointData().GetArray('Scalars_'))
levels = [float(x) for x in np.quantile(arr, [0.2, 0.4, 0.6, 0.8])]
contours = Contour(registrationName='Four equal-volume isocontours', Input=simp)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = levels
contours.ComputeNormals = 1
contours.UpdatePipeline()

# Persist the requested analysis products.
SaveData('/workspace/QMCPACK_simplified_0.04.vti', proxy=simp)
critical_data = servermanager.Fetch(OutputPort(msc, 1))
critical_dims = vtk_to_numpy(critical_data.GetPointData().GetArray('CellDimension'))
critical_vals = vtk_to_numpy(critical_data.GetPointData().GetArray('Scalars_'))
critical_xyz = np.array([critical_data.GetPoint(i) for i in range(critical_data.GetNumberOfPoints())])
type_names = {0: 'minimum', 1: '1-saddle', 2: '2-saddle', 3: 'maximum'}
with open('/workspace/QMCPACK_PL_critical_points_0.04.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['critical_type', 'cell_dimension', 'scalar_value', 'x', 'y', 'z'])
    for dim, val, xyz in zip(critical_dims, critical_vals, critical_xyz):
        writer.writerow([type_names[int(dim)], int(dim), float(val), *map(float, xyz)])

view = CreateView('RenderView')
view.ViewSize = [1500, 1050]
view.Background = [0.07, 0.08, 0.11]
view.OrientationAxesVisibility = 1

cd = Show(contours, view)
cd.Representation = 'Surface'
cd.ColorArrayName = ['POINTS', 'Scalars_']
cd.LookupTable = GetColorTransferFunction('Scalars_')
cd.Opacity = 0.42
cd.Specular = 0.25
cd.SpecularPower = 22

# CellDimension: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
critical_glyphs = Glyph(registrationName='Colored PL critical points', Input=OutputPort(msc, 1), GlyphType='Sphere')
critical_glyphs.GlyphType.Radius = 1.0
critical_glyphs.GlyphType.ThetaResolution = 20
critical_glyphs.GlyphType.PhiResolution = 20
critical_glyphs.ScaleArray = ['POINTS', 'No scale array']
critical_glyphs.ScaleFactor = 0.45
critical_glyphs.GlyphMode = 'All Points'
critical_display = Show(critical_glyphs, view)
critical_display.ColorArrayName = ['POINTS', 'CellDimension']
critical_lut = GetColorTransferFunction('CellDimension')
critical_lut.InterpretValuesAsCategories = 1
critical_lut.Annotations = ['0', 'Minima', '1', '1-saddles', '2', '2-saddles', '3', 'Maxima']
critical_lut.IndexedColors = [0.10,0.35,1.0, 1.0,1.0,1.0, 1.0,0.45,0.02, 1.0,0.03,0.03]
critical_display.LookupTable = critical_lut
critical_display.Specular = 0.5
critical_display.SpecularPower = 30

view.ResetCamera()
view.CameraPosition = [210, -245, 190]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 0
Render(view)
SaveScreenshot('/workspace/QMCPACK_simplified_0.04_visualization.png', view, ImageResolution=[1500,1050])
SaveState('/workspace/QMCPACK_simplified_0.04.pvsm')
print('ISOVALUES', levels)
print('CRITICAL_POINTS_CSV /workspace/QMCPACK_PL_critical_points_0.04.csv')
