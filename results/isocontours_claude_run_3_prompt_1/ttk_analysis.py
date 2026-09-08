from paraview.simple import *
import numpy as np

paraview.simple._DisableFirstRenderCameraReset()

LoadPlugin("/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so", remote=False, ns=globals())

# --- Load data ---
reader = XMLImageDataReader(FileName=["/workspace/QMCPACK.vti"])
reader.UpdatePipeline()

di = servermanager.Fetch(reader)
pd = di.GetPointData()
arr = pd.GetArray(0)
arrname = arr.GetName()
print("Using array:", arrname)

from vtk.util.numpy_support import vtk_to_numpy
vals = vtk_to_numpy(pd.GetArray(arrname))

# 4 isovalues dividing the volume into 5 evenly sized regions (by voxel count / quantiles)
qs = np.percentile(vals, [20, 40, 60, 80])
isovalues = [float(q) for q in qs]
print("Isovalues (5 equal-volume regions):", isovalues)

# --- TTK persistence-based simplification ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = arrname
persistenceDiagram.UpdatePipeline()

threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['POINTS', 'Persistence']
threshold.UpperThreshold = 0.04
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpdatePipeline()

topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
topoSimplification.ScalarField = arrname
topoSimplification.UpdatePipeline()

# --- Critical points on the simplified field ---
morseSmale = TTKMorseSmaleComplex(Input=topoSimplification)
morseSmale.ScalarField = arrname
morseSmale.UpdatePipeline()

critPoints = OutputPort(morseSmale, 0)

# --- Render View ---
view = CreateRenderView()
view.ViewSize = [1400, 1000]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]

# Isocontours: 4 isovalues dividing volume into 5 equal regions
contour = Contour(Input=topoSimplification)
contour.ContourBy = ['POINTS', arrname]
contour.Isosurfaces = isovalues
contour.PointMergeMethod = 'Uniform Binning'
contourDisplay = Show(contour, view)
contourDisplay.Representation = 'Surface'
ColorBy(contourDisplay, ('POINTS', arrname))
contourLUT = GetColorTransferFunction(arrname)
contourLUT.ApplyPreset('Cool to Warm', True)
contourDisplay.SetScalarBarVisibility(view, True)
contourDisplay.Opacity = 0.35

# Critical points colored by CriticalType
cpDisplay = Show(critPoints, view)
cpDisplay.Representation = 'Points'
cpDisplay.PointSize = 12
cpDisplay.RenderPointsAsSpheres = 1
ColorBy(cpDisplay, ('POINTS', 'CellDimension'))

lut = GetColorTransferFunction('CellDimension')
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
# CellDimension: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum (3D scalar field)
lut.Annotations = ['0', 'min', '1', '1-saddle', '2', '2-saddle', '3', 'max']
lut.IndexedColors = [
    0.0, 0.0, 1.0,   # min -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # max -> red
]
cpDisplay.LookupTable = lut

view.ResetCamera()
view.CameraPosition = view.CameraPosition
Render(view)

SaveScreenshot("/workspace/QMCPACK_ttk_visualization.png", view, ImageResolution=[1400, 1000])
print("Saved screenshot to /workspace/QMCPACK_ttk_visualization.png")
