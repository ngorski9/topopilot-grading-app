from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

SCALAR = 'Scalars_'
PERSISTENCE_THRESHOLD = 0.05

# --- Load data ---
reader = XMLImageDataReader(FileName=['fracture.vti'])
reader.PointArrayStatus = [SCALAR]

# --- Persistence diagram of the input field ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', SCALAR]

# --- Keep only persistence pairs above threshold (remove noise) ---
critPairs = Threshold(Input=persistenceDiagram)
critPairs.Scalars = ['CELLS', 'Persistence']
critPairs.LowerThreshold = PERSISTENCE_THRESHOLD
critPairs.UpperThreshold = 1e9
critPairs.ThresholdMethod = 'Above Upper Threshold'

# --- Topological simplification using the surviving pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
simplification.ScalarField = ['POINTS', SCALAR]
simplification.VertexIdentifierField = ['POINTS', 'CriticalPointId']

# --- Morse-Smale complex of the simplified field ---
morseSmale = TTKMorseSmaleComplex(Input=simplification)
morseSmale.ScalarField = ['POINTS', SCALAR]
morseSmale.AscendingSegmentation = 1
morseSmale.DescendingSegmentation = 1
morseSmale.MorseSmaleComplexSegmentation = 1
morseSmale.Ascending1Separatrices = 0
morseSmale.Ascending2Separatrices = 0
morseSmale.Descending1Separatrices = 0
morseSmale.Descending2Separatrices = 0
morseSmale.SaddleConnectors = 0

renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0

# --- Segmentation (piecewise-linear Morse-Smale segmentation) ---
segOutput = OutputPort(morseSmale, 3)  # port 3: Segmentation
segDisplay = Show(segOutput, renderView)
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segDisplay.SetScalarBarVisibility(renderView, True)
segDisplay.RescaleTransferFunctionToDataRange(True)

# --- Critical points ---
critPointsOutput = OutputPort(morseSmale, 0)  # port 0: Critical Points

glyphSpheres = Glyph(Input=critPointsOutput, GlyphType='Sphere')
glyphSpheres.GlyphType.Radius = 2.0
glyphSpheres.ScaleArray = ['POINTS', 'No scale array']
glyphSpheres.ScaleFactor = 1.0
glyphSpheres.GlyphMode = 'All Points'

glyphDisplay = Show(glyphSpheres, renderView, 'GeometryRepresentation')

# CellDimension encodes the critical-point type in 2D fields: 0 = minimum, 1 = saddle, 2 = maximum
lut = GetColorTransferFunction('CellDimension')
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
lut.Annotations = ['0', 'Minimum', '1', 'Saddle', '2', 'Maximum']
lut.IndexedColors = [
    0.0, 0.0, 1.0,   # Minimum -> blue
    1.0, 1.0, 1.0,   # Saddle -> white
    1.0, 0.0, 0.0,   # Maximum -> red
]

ColorBy(glyphDisplay, ('POINTS', 'CellDimension'))
glyphDisplay.SetScalarBarVisibility(renderView, False)

renderView.ResetCamera()
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/fracture_morse_smale.png', renderView, ImageResolution=[1600, 1200])
print('Saved screenshot to /workspace/fracture_morse_smale.png')
