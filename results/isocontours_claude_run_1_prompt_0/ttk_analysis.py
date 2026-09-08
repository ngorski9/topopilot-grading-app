import numpy as np
from paraview.simple import *
from paraview import servermanager as sm

LoadDistributedPlugin('TopologyToolKit', ns=globals())

# --- Load data ---
reader = XMLImageDataReader(FileName=['QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']

# --- Compute quantile isovalues for 5 evenly sized (by voxel count) regions ---
fetched = sm.Fetch(reader)
from vtk.util.numpy_support import vtk_to_numpy
arr = vtk_to_numpy(fetched.GetPointData().GetArray('Scalars_'))
qs = np.percentile(arr, [20, 40, 60, 80])
isovalues = sorted(qs.tolist())
print("Isovalues dividing volume into 5 equal regions:", isovalues)

# --- TTK pipeline: Persistence Diagram -> threshold -> simplification -> Morse-Smale critical points ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']

# Keep only pairs with persistence > 0.04, remove the global min-max pair's endpoints filtering handled by threshold
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.04
threshold.UpperThreshold = 1e10
threshold.ThresholdMethod = 'Between'

simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']

morseSmale = TTKMorseSmaleComplex(Input=simplification)
morseSmale.ScalarField = ['POINTS', 'Scalars_']
morseSmale.CriticalPoints = 1
morseSmale.Ascending1Separatrices = 0
morseSmale.Descending1Separatrices = 0
morseSmale.SaddleConnectors = 0
morseSmale.AscendingSegmentation = 0
morseSmale.DescendingSegmentation = 0
morseSmale.Ascending2Separatrices = 0
morseSmale.Descending2Separatrices = 0

# --- Isocontours ---
contour = Contour(Input=simplification)
contour.ContourBy = ['POINTS', 'Scalars_']
contour.Isosurfaces = isovalues
contour.PointMergeMethod = 'Uniform Binning'

# --- Render ---
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

contourDisplay = Show(contour, view)
contourDisplay.Representation = 'Surface'
contourDisplay.Opacity = 0.25
contourDisplay.DiffuseColor = [0.6, 0.6, 0.6]

cpOutput = OutputPort(morseSmale, 0)  # critical points output port
cpDisplay = Show(cpOutput, view)
cpDisplay.Representation = 'Points'
cpDisplay.RenderPointsAsSpheres = 1
cpDisplay.PointSize = 12

ColorBy(cpDisplay, ('POINTS', 'CriticalType'))
ctf = GetColorTransferFunction('CriticalType')
ctf.InterpretValuesAsCategories = 1
ctf.AnnotationsInitialized = 1
# TTK CriticalType (3D): 0=min, 1=1-saddle, 2=2-saddle, 3=max
ctf.Annotations = ['0', 'minimum', '1', '1-saddle', '2', '2-saddle', '3', 'maximum']
ctf.IndexedColors = [
    0.0, 0.0, 1.0,   # 0 min -> blue
    1.0, 1.0, 1.0,   # 1 1-saddle -> white
    1.0, 0.5, 0.0,   # 2 2-saddle -> orange
    1.0, 0.0, 0.0,   # 3 max -> red
]
cpDisplay.SetScalarBarVisibility(view, True)

view.Background = [0.15, 0.15, 0.15]
ResetCamera(view)
view.CameraViewAngle = 30

Render(view)
SaveScreenshot('/workspace/qmcpack_ttk_visualization.png', view, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/qmcpack_ttk_visualization.png")
