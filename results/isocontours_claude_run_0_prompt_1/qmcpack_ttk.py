from paraview.simple import *
import paraview.servermanager as sm

paraview.simple._DisableFirstRenderCameraReset()

THRESHOLD = 0.04

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

array_name = 'Scalars_'
rng = reader.GetDataInformation().GetPointDataInformation().GetArrayInformation(array_name).GetComponentRange(0)
vmin, vmax = rng[0], rng[1]
print('scalar range', vmin, vmax)

# ---------------------------------------------------------------------------
# 2. Persistence diagram of the raw field
# ---------------------------------------------------------------------------
pdiag = TTKPersistenceDiagram(Input=reader)
pdiag.ScalarField = ['POINTS', array_name]
pdiag.UpdatePipeline()

# ---------------------------------------------------------------------------
# 3. Keep only pairs with persistence >= THRESHOLD (simplification criterion)
# ---------------------------------------------------------------------------
pthresh = Threshold(Input=pdiag)
pthresh.Scalars = ['CELLS', 'Persistence']
pthresh.UpperThreshold = THRESHOLD
pthresh.ThresholdMethod = 'Above Upper Threshold'
pthresh.UpdatePipeline()

# ---------------------------------------------------------------------------
# 4. Topological simplification driven by the surviving persistence pairs
# ---------------------------------------------------------------------------
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=pthresh)
simplify.ScalarField = ['POINTS', array_name]
simplify.VertexIdentifierField = ['POINTS', 'VertexIdentifier']
simplify.UpdatePipeline()

# ---------------------------------------------------------------------------
# 5. Piecewise-linear critical points of the simplified field
# ---------------------------------------------------------------------------
crit = TTKScalarFieldCriticalPoints(Input=simplify)
crit.ScalarField = ['POINTS', array_name]
crit.UpdatePipeline()

ncp = crit.GetDataInformation().GetNumberOfPoints()
print('number of critical points after simplification:', ncp)

# ---------------------------------------------------------------------------
# 6. Four isovalues splitting the scalar range into 5 evenly sized bands
# ---------------------------------------------------------------------------
step = (vmax - vmin) / 5.0
iso_values = [vmin + step * k for k in (1, 2, 3, 4)]
print('isovalues', iso_values)

contour = Contour(Input=simplify)
contour.ContourBy = ['POINTS', array_name]
contour.Isosurfaces = iso_values
contour.PointMergeMethod = 'Uniform Binning'
contour.UpdatePipeline()

# ---------------------------------------------------------------------------
# 7. Render
# ---------------------------------------------------------------------------
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]
view.Background = [0.15, 0.15, 0.16]

# volume outline for context
outline = Outline(Input=reader)
outlineDisp = Show(outline, view)
outlineDisp.AmbientColor = [1, 1, 1]
outlineDisp.DiffuseColor = [1, 1, 1]
outlineDisp.LineWidth = 1.0

# isocontours, semi-transparent, colored by scalar value
contourDisp = Show(contour, view)
ColorBy(contourDisp, ('POINTS', array_name))
contourDisp.SetRepresentationType('Surface')
contourDisp.Opacity = 0.35
contourLUT = GetColorTransferFunction(array_name)
contourLUT.ApplyPreset('Cool to Warm', True)
contourDisp.RescaleTransferFunctionToDataRange(False, [vmin, vmax])
hideBar = GetScalarBar(contourLUT, view)
hideBar.Visibility = 0

# critical points, colored by CriticalType
critDisp = Show(crit, view)
critDisp.SetRepresentationType('Points')
critDisp.RenderPointsAsSpheres = 1
critDisp.PointSize = 14

ColorBy(critDisp, ('POINTS', 'CriticalType'))
ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1
# CriticalType: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
ctLUT.Annotations = ['0', 'minimum', '1', '1-saddle', '2', '2-saddle', '3', 'maximum']
ctLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # minimum -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.55, 0.0,  # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maximum -> red
]
critDisp.RescaleTransferFunctionToDataRange(False, [0, 3])
critBar = GetScalarBar(ctLUT, view)
critBar.Visibility = 1
critBar.Title = 'Critical Type'
critBar.ComponentTitle = ''

ResetCamera(view)
view.CameraPosition = [200, -200, 200]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
ResetCamera(view)
Render(view)

SaveScreenshot('/workspace/qmcpack_ttk_result.png', view, ImageResolution=[1600, 1200])
print('Saved screenshot to /workspace/qmcpack_ttk_result.png')
