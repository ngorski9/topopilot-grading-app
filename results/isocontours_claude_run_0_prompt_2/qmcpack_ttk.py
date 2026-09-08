
from paraview.simple import *
import numpy as np

paraview.simple._DisableFirstRenderCameraReset()

FIELD = "Scalars_"

# 1. Load data
reader = XMLImageDataReader(FileName=['./QMCPACK.vti'])
reader.PointArrayStatus = [FIELD]

# 2. Persistence diagram
pdiag = TTKPersistenceDiagram(Input=reader)
pdiag.ScalarField = ['POINTS', FIELD]

# 3. Keep only pairs with persistence >= 0.04 (drop the global min-max pair's
#    infinite/essential entry is handled automatically by TTK; threshold on Persistence)
pthresh = Threshold(Input=pdiag)
pthresh.Scalars = ['CELLS', 'Persistence']
pthresh.LowerThreshold = 0.04
pthresh.UpperThreshold = 1e9
pthresh.ThresholdMethod = 'Between'

# 4. Topological simplification driven by the simplified diagram
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=pthresh)
simplify.ScalarField = ['POINTS', FIELD]
simplify.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# 5. Critical points of the simplified field (piecewise-linear)
crit = TTKScalarFieldCriticalPoints(Input=simplify)
crit.ScalarField = ['POINTS', FIELD]

crit.UpdatePipeline()
critData = servermanager.Fetch(crit)

# 6. Compute isovalues for 4 isosurfaces splitting volume into 5 equal-sized regions
simplify.UpdatePipeline()
simpData = servermanager.Fetch(simplify)
from vtk.util.numpy_support import vtk_to_numpy
arr = vtk_to_numpy(simpData.GetPointData().GetArray(FIELD))
qs = [20, 40, 60, 80]
isovalues = list(np.percentile(arr, qs))
print("Isovalues (5 equal-volume regions):", isovalues)

# 7. Render setup
renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0
renderView.Background = [1, 1, 1]

# Show simplified volume outline for context
simpDisplay = Show(simplify, renderView)
simpDisplay.Representation = 'Outline'
simpDisplay.AmbientColor = [0.6, 0.6, 0.6]
simpDisplay.DiffuseColor = [0.6, 0.6, 0.6]

# Isocontours
contour = Contour(Input=simplify)
contour.ContourBy = ['POINTS', FIELD]
contour.Isosurfaces = isovalues
contour.ComputeScalars = 1

contourDisplay = Show(contour, renderView)
ColorBy(contourDisplay, ('POINTS', FIELD))
contourDisplay.SetScalarBarVisibility(renderView, True)
contourDisplay.Opacity = 0.35
contour_LUT = GetColorTransferFunction(FIELD)
contour_LUT.ApplyPreset('Cool to Warm', True)

# Critical points colored by type
critDisplay = Show(crit, renderView)
critDisplay.Representation = 'Points'
critDisplay.PointSize = 12
critDisplay.RenderPointsAsSpheres = 1

ColorBy(critDisplay, ('POINTS', 'CriticalType'))
ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1

# CriticalType values: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum (3D scalar field)
ctLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.55, 0.0,  # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]
critDisplay.SetScalarBarVisibility(renderView, True)

renderView.ResetCamera()
renderView.CameraPosition = [200, -250, 200]
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/qmcpack_critical_points.png', renderView, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/qmcpack_critical_points.png")

# Print critical point summary
ctArr = vtk_to_numpy(critData.GetPointData().GetArray('CriticalType'))
names = {0: 'minima', 1: '1-saddles', 2: '2-saddles', 3: 'maxima'}
for k, v in names.items():
    print(f"{v}: {(ctArr == k).sum()}")
