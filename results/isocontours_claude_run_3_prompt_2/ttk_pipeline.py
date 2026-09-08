from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())

SCALAR = 'Scalars_'
PERSISTENCE_THRESHOLD = 0.04

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

# --- Compute persistence diagram on the raw field ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', SCALAR]
persistenceDiagram.UpdatePipeline()

# --- Keep only pairs with persistence >= threshold (simplification criteria) ---
critPairs = Threshold(Input=persistenceDiagram)
critPairs.Scalars = ['CELLS', 'Persistence']
critPairs.UpperThreshold = PERSISTENCE_THRESHOLD
critPairs.ThresholdMethod = 'Above Upper Threshold'
critPairs.UpdatePipeline()

# --- Topological simplification using the surviving pairs as constraints ---
topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
topoSimplification.ScalarField = ['POINTS', SCALAR]
topoSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
topoSimplification.UpdatePipeline()

# --- Extract piecewise-linear critical points of the simplified field ---
criticalPoints = TTKScalarFieldCriticalPoints(Input=topoSimplification)
criticalPoints.ScalarField = ['POINTS', SCALAR]
criticalPoints.UpdatePipeline()

# --- Isocontours: 4 isosurfaces dividing the scalar range into 5 equal regions ---
rng = reader.PointData[SCALAR].GetRange()
lo, hi = rng
step = (hi - lo) / 5.0
isovalues = [lo + step, lo + 2 * step, lo + 3 * step, lo + 4 * step]

contour = Contour(Input=topoSimplification)
contour.ContourBy = ['POINTS', SCALAR]
contour.Isosurfaces = isovalues
contour.UpdatePipeline()

# ============ RENDERING ============
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]
view.OrientationAxesVisibility = 0
view.Background = [1.0, 1.0, 1.0]

# Contour surfaces: translucent grey-ish shading to show the 5 regions
contourDisplay = Show(contour, view, 'GeometryRepresentation')
ColorBy(contourDisplay, ('POINTS', SCALAR))
contourDisplay.SetScalarBarVisibility(view, False)
contourLUT = GetColorTransferFunction(SCALAR)
contourLUT.ApplyPreset('Cool to Warm', True)
contourDisplay.Opacity = 0.35

# Critical points, colored categorically by CriticalType
cpDisplay = Show(criticalPoints, view, 'GeometryRepresentation')
cpDisplay.SetRepresentationType('Points')
cpDisplay.PointSize = 14
cpDisplay.RenderPointsAsSpheres = 1

ColorBy(cpDisplay, ('POINTS', 'CriticalType'))
ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1
# CriticalType: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum (3D scalar field)
ctLUT.Annotations = ['0', 'Minima', '1', '1-Saddles', '2', '2-Saddles', '3', 'Maxima']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # 0 minima -> blue
    1.0, 1.0, 1.0,   # 1 1-saddles -> white
    1.0, 0.5, 0.0,   # 2 2-saddles -> orange
    1.0, 0.0, 0.0,   # 3 maxima -> red
]
cpDisplay.SetScalarBarVisibility(view, True)
sb = GetScalarBar(ctLUT, view)
sb.Title = 'Critical Type'
sb.ComponentTitle = ''

ResetCamera(view)
view.CameraPosition = [200, -200, 150]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
ResetCamera(view)
view.Update()

Render(view)
SaveScreenshot('/workspace/qmcpack_ttk_critical_points.png', view, ImageResolution=[1600, 1200])
print('Saved screenshot to /workspace/qmcpack_ttk_critical_points.png')
print('Scalar range:', rng)
print('Isovalues:', isovalues)

# Report critical point counts by type
import paraview.servermanager as sm
data = sm.Fetch(criticalPoints)
ct_array = data.GetPointData().GetArray('CriticalType')
counts = {}
if ct_array is not None:
    for i in range(ct_array.GetNumberOfTuples()):
        v = int(ct_array.GetTuple1(i))
        counts[v] = counts.get(v, 0) + 1
print('Critical point counts by type:', counts)
