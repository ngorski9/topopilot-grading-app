from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

array_name = "Scalars_"
persistence_threshold = 0.05
critical_point_radius = 2.0

# --- Load data ---
reader = XMLImageDataReader(FileName=["/workspace/fracture.vti"])
reader.PointArrayStatus = [array_name]

# --- Persistence diagram of the raw field ---
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ['POINTS', array_name]

# --- Keep only persistence pairs above the simplification threshold ---
pdThresh = Threshold(Input=pd)
pdThresh.Scalars = ['CELLS', 'Persistence']
pdThresh.ThresholdMethod = 'Between'
pdThresh.LowerThreshold = persistence_threshold
pdThresh.UpperThreshold = 1e9

# --- Topologically simplify the scalar field using the surviving pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=pdThresh)
simplification.ScalarField = ['POINTS', array_name]
simplification.VertexIdentifierField = ['POINTS', 'VertexIdentifier']

# --- Morse-Smale complex of the simplified field ---
msc = TTKMorseSmaleComplex(Input=simplification)
msc.ScalarField = ['POINTS', array_name]
msc.UpdatePipeline()

criticalPoints = OutputPort(msc, 0)   # critical points
segmentation = OutputPort(msc, 3)     # piecewise-linear MS segmentation

view = GetActiveViewOrCreate('RenderView')
view.OrientationAxesVisibility = 0

# --- Segmentation surface, colored by Morse-Smale region id ---
segDisplay = Show(segmentation, view, 'GeometryRepresentation')
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Rainbow Uniform', True)
segDisplay.SetScalarBarVisibility(view, False)

# --- Critical points as spheres, colored by type ---
glyph = Glyph(Input=criticalPoints, GlyphType='Sphere')
glyph.GlyphType.Radius = critical_point_radius
glyph.OrientationArray = ['POINTS', 'No orientation array']
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

glyphDisplay = Show(glyph, view, 'GeometryRepresentation')
glyphDisplay.Representation = 'Surface'
ColorBy(glyphDisplay, ('POINTS', 'CellDimension'))

# CellDimension: 0 = minimum, 1 = saddle, 2 = maximum (for a 2D scalar field)
critLUT = GetColorTransferFunction('CellDimension')
critLUT.InterpretValuesAsCategories = 1
critLUT.AnnotationsInitialized = 1
critLUT.Annotations = ['0', 'Minimum', '1', 'Saddle', '2', 'Maximum']
critLUT.IndexedColors = [
    0.0, 0.0, 1.0,   # Minimum -> blue
    1.0, 1.0, 1.0,   # Saddle -> white
    1.0, 0.0, 0.0,   # Maximum -> red
]
glyphDisplay.SetScalarBarVisibility(view, False)

view.ResetCamera()
view.Background = [0.0, 0.0, 0.0]
Render(view)

SaveScreenshot('/workspace/fracture_msc.png', view, ImageResolution=[1600, 900])
print("Saved /workspace/fracture_msc.png")
