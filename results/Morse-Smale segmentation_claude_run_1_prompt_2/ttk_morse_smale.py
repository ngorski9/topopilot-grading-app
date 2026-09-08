from paraview.simple import *

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.UpdatePipeline()

array_name = 'Scalars_'

# --- Persistence diagram ---
persDiag = TTKPersistenceDiagram(Input=reader)
persDiag.ScalarField = ['POINTS', array_name]
persDiag.UpdatePipeline()

# --- Threshold the diagram pairs by persistence (keep pairs with persistence >= 0.05) ---
threshold = Threshold(Input=persDiag)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.05
threshold.UpperThreshold = 1e10
threshold.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'
threshold.UpdatePipeline()

# --- Topological simplification using the simplified set of critical pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', array_name]
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

# --- Morse-Smale complex on the simplified scalar field ---
msc = TTKMorseSmaleComplex(Input=simplification)
msc.ScalarField = ['POINTS', array_name]
msc.CriticalPoints = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1
msc.Ascending2Separatrices = 1
msc.Descending2Separatrices = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.MorseSmaleComplexSegmentation = 1
msc.UpdatePipeline()

critPointsSrc = OutputPort(msc, 0)
segmentationSrc = OutputPort(msc, 3)

renderView = GetActiveViewOrCreate('RenderView')

# --- Show segmentation: the piecewise-linear Morse-Smale segmentation ---
segDisplay = Show(segmentationSrc, renderView)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Rainbow Desaturated', True)
segDisplay.SetScalarBarVisibility(renderView, False)
segDisplay.RescaleTransferFunctionToDataRange(True)

# --- Critical points: in a 2D field, CellDimension gives the type:
#     0 = minimum, 1 = saddle, 2 = maximum ---
minima = Threshold(Input=critPointsSrc)
minima.Scalars = ['POINTS', 'CellDimension']
minima.LowerThreshold = 0
minima.UpperThreshold = 0
minima.ThresholdMethod = 'Between'

maxima = Threshold(Input=critPointsSrc)
maxima.Scalars = ['POINTS', 'CellDimension']
maxima.LowerThreshold = 2
maxima.UpperThreshold = 2
maxima.ThresholdMethod = 'Between'

saddles = Threshold(Input=critPointsSrc)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1
saddles.ThresholdMethod = 'Between'

def show_glyph(src, color, name):
    Show(src, renderView)
    Hide(src, renderView)
    glyph = Glyph(Input=src, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    disp = Show(glyph, renderView)
    disp.Representation = 'Surface'
    ColorBy(disp, None)
    disp.AmbientColor = color
    disp.DiffuseColor = color
    RenameSource(name, glyph)
    return glyph

minGlyph = show_glyph(minima, [0.0, 0.0, 1.0], 'Minima')
maxGlyph = show_glyph(maxima, [1.0, 0.0, 0.0], 'Maxima')
saddleGlyph = show_glyph(saddles, [1.0, 1.0, 1.0], 'Saddles')

renderView.ResetCamera()
renderView.Background = [0.32, 0.34, 0.43]
Render(renderView)

SaveScreenshot('/workspace/morse_smale_result.png', renderView, ImageResolution=[1600, 1000])
print('Saved /workspace/morse_smale_result.png')
