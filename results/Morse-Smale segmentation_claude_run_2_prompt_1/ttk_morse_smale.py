from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()

reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

# TTK requires an explicit triangulation-friendly dataset; TetrahedralizeCommand not needed for 2D image data,
# but TTK filters need a vtkUnstructuredGrid/vtkPolyData style triangulation for PL Morse-Smale on 2D grids.
tetrahedralize = Tetrahedralize(Input=reader)

persistenceDiagram = TTKPersistenceDiagram(Input=tetrahedralize)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']

# Threshold the persistence diagram to keep pairs with persistence >= 0.05
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.05
threshold.UpperThreshold = 1e10
threshold.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'

topologicalSimplification = TTKTopologicalSimplification(Domain=tetrahedralize, Constraints=threshold)
topologicalSimplification.ScalarField = ['POINTS', 'Scalars_']
topologicalSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

morseSmale = TTKMorseSmaleComplex(Input=topologicalSimplification)
morseSmale.ScalarField = ['POINTS', 'Scalars_']
morseSmale.CriticalPoints = 1
morseSmale.Ascending1Separatrices = 0
morseSmale.Descending1Separatrices = 0
morseSmale.Ascending2Separatrices = 0
morseSmale.Descending2Separatrices = 0
morseSmale.SaddleConnectors = 0
morseSmale.AscendingSegmentation = 1
morseSmale.DescendingSegmentation = 1
morseSmale.MorseSmaleComplexSegmentation = 1

renderView1 = GetActiveViewOrCreate('RenderView')

segmentation = OutputPort(morseSmale, 3)
segDisplay = Show(segmentation, renderView1)
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segDisplay.SetRepresentationType('Surface')
segDisplay.RescaleTransferFunctionToDataRange(True)

# Critical points output is OutputPort 0
critPoints = OutputPort(morseSmale, 0)

# Split critical points by CellDimension: 0=min,1=saddle1,2=saddle2(max in 2D),3=max (depends on dim); for 2D field, types are 0=min,1=saddle,3=max
mins = Threshold(Input=critPoints)
mins.Scalars = ['POINTS', 'CellDimension']
mins.LowerThreshold = 0
mins.UpperThreshold = 0
mins.ThresholdMethod = 'Between'

saddles = Threshold(Input=critPoints)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1
saddles.ThresholdMethod = 'Between'

maxs = Threshold(Input=critPoints)
maxs.Scalars = ['POINTS', 'CellDimension']
maxs.LowerThreshold = 2
maxs.UpperThreshold = 3
maxs.ThresholdMethod = 'Between'

minsDisplay = Show(mins, renderView1)
minsDisplay.SetRepresentationType('Points')
minsDisplay.ColorArrayName = ['POINTS', '']
minsDisplay.PointSize = 12
minsDisplay.RenderPointsAsSpheres = 1
minsDisplay.AmbientColor = [0.0, 0.0, 1.0]
minsDisplay.DiffuseColor = [0.0, 0.0, 1.0]

saddlesDisplay = Show(saddles, renderView1)
saddlesDisplay.SetRepresentationType('Points')
saddlesDisplay.ColorArrayName = ['POINTS', '']
saddlesDisplay.PointSize = 12
saddlesDisplay.RenderPointsAsSpheres = 1
saddlesDisplay.AmbientColor = [1.0, 1.0, 1.0]
saddlesDisplay.DiffuseColor = [1.0, 1.0, 1.0]

maxsDisplay = Show(maxs, renderView1)
maxsDisplay.SetRepresentationType('Points')
maxsDisplay.ColorArrayName = ['POINTS', '']
maxsDisplay.PointSize = 12
maxsDisplay.RenderPointsAsSpheres = 1
maxsDisplay.AmbientColor = [1.0, 0.0, 0.0]
maxsDisplay.DiffuseColor = [1.0, 0.0, 0.0]

renderView1.ResetCamera()
renderView1.OrientationAxesVisibility = 0
Render()

SaveScreenshot('/workspace/morse_smale_segmentation.png', renderView1, ImageResolution=[1600, 900])
print("DONE")
