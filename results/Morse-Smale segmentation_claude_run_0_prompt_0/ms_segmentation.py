from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.UpdatePipeline()

pdi = reader.GetPointDataInformation()
array_name = pdi.GetArray(0).Name
print("Using scalar array:", array_name)

# --- Persistence diagram of the input field ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', array_name]
persistenceDiagram.UpdatePipeline()

# --- Keep only pairs with persistence >= 0.05 ---
persistenceThreshold = Threshold(Input=persistenceDiagram)
persistenceThreshold.Scalars = ['CELLS', 'Persistence']
persistenceThreshold.LowerThreshold = 0.05
persistenceThreshold.UpperThreshold = 1e10
persistenceThreshold.ThresholdMethod = 'Between'
persistenceThreshold.UpdatePipeline()

# --- Topological simplification using the simplified set of pairs ---
topologicalSimplification = TTKTopologicalSimplification(
    Domain=reader, Constraints=persistenceThreshold)
topologicalSimplification.ScalarField = ['POINTS', array_name]
topologicalSimplification.VertexIdentifierField = ['POINTS', 'VertexId']
topologicalSimplification.UpdatePipeline()

# --- Morse-Smale complex of the simplified field (PL segmentation + critical points) ---
morseSmaleComplex = TTKMorseSmaleComplex(Input=topologicalSimplification)
morseSmaleComplex.ScalarField = ['POINTS', array_name]
morseSmaleComplex.CriticalPoints = 1
morseSmaleComplex.AscendingSegmentation = 1
morseSmaleComplex.DescendingSegmentation = 1
morseSmaleComplex.MorseSmaleComplexSegmentation = 1
morseSmaleComplex.Ascending1Separatrices = 0
morseSmaleComplex.Descending1Separatrices = 0
morseSmaleComplex.Ascending2Separatrices = 0
morseSmaleComplex.Descending2Separatrices = 0
morseSmaleComplex.SaddleConnectors = 0
morseSmaleComplex.UpdatePipeline()

renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0

# --- Segmentation (piecewise-linear Morse-Smale segmentation): output port 3 ---
segmentation = OutputPort(morseSmaleComplex, 3)
segDisplay = Show(segmentation, renderView)
segDisplay.SetRepresentationType('Surface')

ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Rainbow Uniform', True)
segDisplay.SetScalarBarVisibility(renderView, False)

# --- Critical points: output port 0 ---
critPoints = OutputPort(morseSmaleComplex, 0)

glyph = Glyph(Input=critPoints, GlyphType='Sphere')
glyph.GlyphType.Radius = 2
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'
glyph.UpdatePipeline()

glyphDisplay = Show(glyph, renderView)
glyphDisplay.SetRepresentationType('Surface')

# Color critical points by CellDimension: 0=minimum, 1=1-saddle, 2=2-saddle/maximum (2D) or maximum (3D)
ColorBy(glyphDisplay, ('POINTS', 'CellDimension'))
ctLUT = GetColorTransferFunction('CellDimension')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1

critPoints.UpdatePipeline()
pdinfo = critPoints.GetPointDataInformation()
critInfo = pdinfo['CellDimension']
crange = critInfo.GetComponentRange(0)
print("CellDimension range:", crange)

# Build categorical annotations/colors depending on dimension present
annotations = []
indexed_colors = []
type_names = {0: 'Minimum', 1: 'Saddle', 2: 'Maximum', 3: 'Maximum'}
color_map = {
    0: [0.0, 0.0, 1.0],   # minima -> blue
    1: [1.0, 1.0, 1.0],   # saddles -> white
    2: [1.0, 0.0, 0.0],   # maxima (2D) -> red
    3: [1.0, 0.0, 0.0],   # maxima (3D) -> red
}

lo, hi = int(crange[0]), int(crange[1])
for v in range(lo, hi + 1):
    annotations += [str(v), type_names.get(v, str(v))]
    indexed_colors += color_map.get(v, [0.5, 0.5, 0.5])

ctLUT.Annotations = annotations
ctLUT.IndexedColors = indexed_colors
glyphDisplay.SetScalarBarVisibility(renderView, False)

renderView.ResetCamera()
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/ms_segmentation.png', renderView, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/ms_segmentation.png")
