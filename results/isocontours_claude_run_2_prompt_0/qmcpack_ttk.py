from paraview.simple import *
import numpy as np

paraview.simple._DisableFirstRenderCameraReset()

reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()
arrayName = 'Scalars_'

# ---- compute 4 isovalues that split volume into 5 equal-count (equal-volume, since uniform grid) bins
from paraview import servermanager as sm
data = sm.Fetch(reader)
from vtk.numpy_interface import dataset_adapter as dsa
wdata = dsa.WrapDataObject(data)
vals = np.asarray(wdata.PointData[arrayName]).ravel()
percentiles = [20, 40, 60, 80]
isovalues = [float(np.percentile(vals, p)) for p in percentiles]
print("Isovalues (5 equal-volume regions):", isovalues)

# ---- TTK persistence simplification pipeline
pd1 = TTKPersistenceDiagram(Input=reader)
pd1.ScalarField = ['POINTS', arrayName]
pd1.UpdatePipeline()

thresh = Threshold(Input=pd1)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.LowerThreshold = 0.04
thresh.UpperThreshold = 1e12
thresh.ThresholdMethod = 'Between'
thresh.UpdatePipeline()

simplify = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simplify.ScalarField = ['POINTS', arrayName]
simplify.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplify.UpdatePipeline()

# ---- critical points of the simplified field (piecewise-linear)
crit = TTKPersistenceDiagram(Input=simplify)
crit.ScalarField = ['POINTS', arrayName]
crit.UpdatePipeline()

# Better: use dedicated critical points from Morse-Smale complex for PL critical points
msc = TTKMorseSmaleComplex(Input=simplify)
msc.ScalarField = ['POINTS', arrayName]
msc.UpdatePipeline()

critPoints = MergeBlocks(Input=OutputPort(msc, 0))
critPoints.UpdatePipeline()

# ---- isocontours
contour = Contour(Input=simplify)
contour.ContourBy = ['POINTS', arrayName]
contour.Isosurfaces = isovalues
contour.UpdatePipeline()

# ---- render
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

contourDisp = Show(contour, view)
contourDisp.Representation = 'Surface'
contourDisp.Opacity = 0.25
contourDisp.DiffuseColor = [0.6, 0.6, 0.6]

critDisp = Show(critPoints, view)
critDisp.Representation = 'Points'
critDisp.PointSize = 10
critDisp.RenderPointsAsSpheres = 1

ColorBy(critDisp, ('POINTS', 'CriticalType'))
lut = GetColorTransferFunction('CriticalType')
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
# CriticalType: 0=min, 1=1-saddle, 2=2-saddle, 3=max (3D scalar field)
lut.Annotations = ['0', 'min', '1', '1-saddle', '2', '2-saddle', '3', 'max']
lut.IndexedColors = [0.0, 0.0, 1.0,   # min -> blue
                      1.0, 1.0, 1.0,   # 1-saddle -> white
                      1.0, 0.5, 0.0,   # 2-saddle -> orange
                      1.0, 0.0, 0.0]   # max -> red
critDisp.SetScalarBarVisibility(view, True)

view.ResetCamera()
view.OrientationAxesVisibility = 0
Render(view)
SaveScreenshot('/workspace/qmcpack_critical_points.png', view, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/qmcpack_critical_points.png")
print("Isovalues used:", isovalues)
