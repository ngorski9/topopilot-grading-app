from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

infile = '/workspace/fracture.vti'
array = 'Scalars_'
persistence_threshold = 0.05

reader = XMLImageDataReader(FileName=[infile])
reader.PointArrayStatus = [array]

# TTK needs a 3D-looking dataset for some filters but 2D image data works directly.

# 1. Compute persistence diagram of the input field
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ['POINTS', array]
pd.InputOffsetField = ['POINTS', array]

# 2. Keep only pairs with persistence above threshold (simplification pairs)
thresh = Threshold(Input=pd)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.ThresholdMethod = 'Between'
thresh.LowerThreshold = 0.0
thresh.UpperThreshold = persistence_threshold

# 3. Topological simplification using those low-persistence pairs
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simplify.ScalarField = ['POINTS', array]
simplify.InputOffsetField = ['POINTS', array]

# 4. Morse-Smale complex of the simplified field (piecewise-linear -> use default PL, not preprocess as simplicial trick needed)
msc = TTKMorseSmaleComplex(Input=simplify)
msc.ScalarField = ['POINTS', array]
msc.OffsetField = ['POINTS', array]
msc.CriticalPoints = 1
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.MorseSmaleComplexSegmentation = 1
msc.SaddleConnectors = 0
msc.Ascending1Separatrices = 0
msc.Descending1Separatrices = 0
msc.Ascending2Separatrices = 0
msc.Descending2Separatrices = 0

msc.UpdatePipeline()

# TTKMorseSmaleComplex has 4 output ports:
# 0=Critical Points, 1=1-Separatrices, 2=2-Separatrices, 3=Segmentation
crit_extract = OutputPort(msc, 0)
seg_extract = OutputPort(msc, 3)

renderView = GetActiveViewOrCreate('RenderView')

# --- Segmentation display: colored by SegmentationId (piecewise-linear MS segmentation) ---
segDisplay = Show(seg_extract, renderView)
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segDisplay.RescaleTransferFunctionToDataRange(True)
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Rainbow Uniform', True)
segDisplay.SetScalarBarVisibility(renderView, False)

# --- Critical points display ---
# CellDimension on critical points: 0=minimum, 1=saddle, 2=maximum (2D scalar field).
# Split into one source per type and assign a solid color to each, to sidestep any
# categorical-LUT/annotation quirks.
critical_types = [
    (0, 'Minima', (0.0, 0.0, 1.0)),
    (1, 'Saddles', (1.0, 1.0, 1.0)),
    (2, 'Maxima', (1.0, 0.0, 0.0)),
]

for dim_value, label, rgb in critical_types:
    typeThresh = Threshold(Input=crit_extract)
    typeThresh.Scalars = ['POINTS', 'CellDimension']
    typeThresh.ThresholdMethod = 'Between'
    typeThresh.LowerThreshold = dim_value
    typeThresh.UpperThreshold = dim_value

    glyph = Glyph(Input=typeThresh, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'

    glyphDisplay = Show(glyph, renderView)
    glyphDisplay.Representation = 'Surface'
    # Lift critical point glyphs slightly off the segmentation surface to avoid z-fighting
    glyphDisplay.Translation = [0.0, 0.0, 1.0]
    ColorBy(glyphDisplay, None)
    glyphDisplay.AmbientColor = list(rgb)
    glyphDisplay.DiffuseColor = list(rgb)
    glyphDisplay.SetScalarBarVisibility(renderView, False)

renderView.ResetCamera()
renderView.Background = [0.2, 0.2, 0.2]
Render(renderView)

SaveScreenshot('/workspace/fracture_msc.png', renderView, ImageResolution=[1600, 900])
print('Saved /workspace/fracture_msc.png')
