from paraview.simple import *
import numpy as np

ARRAY = 'Scalars_'

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

# --- Persistence diagram ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ARRAY
persistenceDiagram.UpdatePipeline()

# --- Threshold the diagram to keep only pairs with persistence >= 0.04 ---
persistenceThreshold = Threshold(Input=persistenceDiagram)
persistenceThreshold.Scalars = ['CELLS', 'Persistence']
persistenceThreshold.UpperThreshold = 0.04
persistenceThreshold.ThresholdMethod = 'Above Upper Threshold'
persistenceThreshold.UpdatePipeline()

# --- Topological simplification using the surviving persistence pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=persistenceThreshold)
simplification.ScalarField = ARRAY
simplification.UpdatePipeline()

# --- Morse-Smale complex on the simplified field to extract critical points ---
morseSmaleComplex = TTKMorseSmaleComplex(Input=simplification)
morseSmaleComplex.ScalarField = ARRAY
morseSmaleComplex.Ascending1Separatrices = 0
morseSmaleComplex.Descending1Separatrices = 0
morseSmaleComplex.Ascending2Separatrices = 0
morseSmaleComplex.Descending2Separatrices = 0
morseSmaleComplex.AscendingSegmentation = 0
morseSmaleComplex.DescendingSegmentation = 0
morseSmaleComplex.MorseSmaleComplexSegmentation = 0
morseSmaleComplex.SaddleConnectors = 0
morseSmaleComplex.UpdatePipeline()

# --- Compute isovalues for 4 isocontours splitting volume into 5 equal-count regions ---
simplified_data = servermanager.Fetch(simplification)
from vtk.numpy_interface import dataset_adapter as dsa
wrapped = dsa.WrapDataObject(simplified_data)
scalars = np.asarray(wrapped.PointData[ARRAY]).ravel()
percentiles = [20, 40, 60, 80]
isovalues = [float(np.percentile(scalars, p)) for p in percentiles]
print("Equal-volume isovalues:", isovalues)

contour = Contour(Input=simplification)
contour.ContourBy = ['POINTS', ARRAY]
contour.Isosurfaces = isovalues
contour.UpdatePipeline()

# ================= Rendering =================
renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1400, 1000]
renderView.Background = [1.0, 1.0, 1.0]
renderView.OrientationAxesVisibility = 0

# Isocontours: translucent grey surfaces to show the 5 regions
contourDisplay = Show(contour, renderView)
contourDisplay.Representation = 'Surface'
contourDisplay.ColorArrayName = [None, '']
contourDisplay.DiffuseColor = [0.7, 0.7, 0.7]
contourDisplay.Opacity = 0.25

# Critical points, port 0 of MorseSmaleComplex output
cpSource = OutputPort(morseSmaleComplex, 0)
cpDisplay = Show(cpSource, renderView)
cpDisplay.Representation = 'Points'
cpDisplay.PointSize = 12
cpDisplay.RenderPointsAsSpheres = 1

# Color by CellDimension: 0=min,1=1-saddle,2=2-saddle,3=max (3D scalar field)
ColorBy(cpDisplay, ('POINTS', 'CellDimension'))
ctf = GetColorTransferFunction('CellDimension')
ctf.InterpretValuesAsCategories = 1
ctf.AnnotationsInitialized = 1
ctf.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
ctf.IndexedColors = [
    0.0, 0.0, 1.0,   # Minimum -> blue
    1.0, 1.0, 1.0,   # 1-Saddle -> white
    1.0, 0.5, 0.0,   # 2-Saddle -> orange
    1.0, 0.0, 0.0,   # Maximum -> red
]
cpDisplay.SetScalarBarVisibility(renderView, True)

renderView.ResetCamera()
renderView.CameraViewAngle = 30
Render(renderView)
SaveScreenshot('/workspace/QMCPACK_ttk_render.png', renderView, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/QMCPACK_ttk_render.png")
