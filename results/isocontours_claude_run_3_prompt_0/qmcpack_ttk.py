from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']

array_name = 'Scalars_'

# --- Persistence Diagram ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', array_name]
persistenceDiagram.IgnoreBoundary = False

# --- Threshold the persistence diagram pairs by persistence >= 0.04 ---
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.UpperThreshold = 0.04
threshold.ThresholdMethod = 'Above Upper Threshold'

# --- Topological Simplification using the simplified diagram ---
topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
topoSimplification.ScalarField = ['POINTS', array_name]
topoSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# --- Critical points (piecewise-linear) on the simplified scalar field ---
criticalPoints = TTKScalarFieldCriticalPoints(Input=topoSimplification)
criticalPoints.ScalarField = ['POINTS', array_name]

# --- 4 isocontours dividing volume into 5 evenly sized (scalar-range) regions ---
lo, hi = 0.0, 1.0
step = (hi - lo) / 5.0
isovalues = [lo + step * i for i in range(1, 5)]

contour = Contour(Input=topoSimplification)
contour.ContourBy = ['POINTS', array_name]
contour.Isosurfaces = isovalues
contour.PointMergeMethod = 'Uniform Binning'

# --- Render view ---
renderView = GetActiveViewOrCreate('RenderView')

# Contours - translucent grey surfaces
contourDisplay = Show(contour, renderView, 'GeometryRepresentation')
contourDisplay.Representation = 'Surface'
contourDisplay.ColorArrayName = [None, '']
contourDisplay.DiffuseColor = [0.7, 0.7, 0.7]
contourDisplay.Opacity = 0.15

# Critical points - color by CriticalType
cpDisplay = Show(criticalPoints, renderView, 'GeometryRepresentation')
cpDisplay.Representation = 'Points'
cpDisplay.RenderPointsAsSpheres = True
cpDisplay.PointSize = 12

ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1
# CriticalType (3D): 0=min, 1=1-saddle, 2=2-saddle, 3=max
ctLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minima -> blue
    1.0, 1.0, 1.0,   # 1-saddles -> white
    1.0, 0.5, 0.0,   # 2-saddles -> orange
    1.0, 0.0, 0.0,   # maxima -> red
]

ColorBy(cpDisplay, ('POINTS', 'CriticalType'))
cpDisplay.SetScalarBarVisibility(renderView, True)

renderView.ResetCamera()
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/qmcpack_critical_points.png', renderView, ImageResolution=[1600, 1200])
print("Isovalues used:", isovalues)
print("Done.")
