from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = ['Scalars_']

# --- Persistence diagram of the raw field ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']

# --- Keep only pairs with persistence >= 0.05 ---
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.05
threshold.UpperThreshold = 1e9
threshold.ThresholdMethod = 'Between'

# --- Topological simplification driven by the surviving persistence pairs ---
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'CriticalPointId']

# --- Morse-Smale complex (piecewise-linear) on the simplified field ---
msc = TTKMorseSmaleComplex(Input=simplification)
msc.ScalarField = ['POINTS', 'Scalars_']
msc.CriticalPoints = 1
msc.Ascending1Separatrices = 1
msc.Descending1Separatrices = 1
msc.Ascending2Separatrices = 0
msc.Descending2Separatrices = 0
msc.AscendingSegmentation = 1
msc.DescendingSegmentation = 1
msc.MorseSmaleComplexSegmentation = 1

msc.UpdatePipeline()

renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0
renderView.UseColorPaletteForBackground = 0
renderView.Background = [1, 1, 1]
renderView.ViewSize = [1600, 900]

# --- Segmentation display (colored by Morse-Smale region id) ---
segmentation = OutputPort(msc, 3)
segDisplay = Show(segmentation, renderView, 'UnstructuredGridRepresentation')
segDisplay.Representation = 'Surface'
ColorBy(segDisplay, ('POINTS', 'MorseSmaleManifold'))
segLUT = GetColorTransferFunction('MorseSmaleManifold')
segLUT.ApplyPreset('Rainbow Desaturated', True)
HideScalarBarIfNotNeeded(segLUT, renderView)

# --- Critical points ---
# CellDimension: 0 = minimum, 1 = saddle, 2 = maximum (2D piecewise-linear field)
criticalPoints = OutputPort(msc, 0)

mins = Threshold(Input=criticalPoints)
mins.Scalars = ['POINTS', 'CellDimension']
mins.LowerThreshold = 0
mins.UpperThreshold = 0
mins.ThresholdMethod = 'Between'

saddles = Threshold(Input=criticalPoints)
saddles.Scalars = ['POINTS', 'CellDimension']
saddles.LowerThreshold = 1
saddles.UpperThreshold = 1
saddles.ThresholdMethod = 'Between'

maxs = Threshold(Input=criticalPoints)
maxs.Scalars = ['POINTS', 'CellDimension']
maxs.LowerThreshold = 2
maxs.UpperThreshold = 2
maxs.ThresholdMethod = 'Between'

def show_points(src, color, radius, label):
    src.UpdatePipeline()
    n = src.GetDataInformation().GetNumberOfPoints()
    print(label, 'count =', n)
    glyph = Glyph(Input=src, GlyphType='Sphere')
    glyph.GlyphType.Radius = radius
    glyph.GlyphType.ThetaResolution = 16
    glyph.GlyphType.PhiResolution = 16
    glyph.ScaleFactor = 1.0
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.GlyphMode = 'All Points'
    disp = Show(glyph, renderView, 'GeometryRepresentation')
    disp.Representation = 'Surface'
    ColorBy(disp, None)
    disp.DiffuseColor = color
    disp.AmbientColor = color
    disp.SpecularColor = color
    return glyph, disp

show_points(mins, [0.0, 0.0, 1.0], 2.0, 'mins')      # blue mins
show_points(maxs, [1.0, 0.0, 0.0], 2.0, 'maxs')      # red maxes
show_points(saddles, [1.0, 1.0, 1.0], 2.0, 'saddles')   # white saddles

renderView.ResetCamera()
renderView.CameraParallelProjection = 1
Render()

SaveScreenshot('/workspace/morse_smale_segmentation.png', renderView, ImageResolution=[1600, 900])
print("Saved screenshot to /workspace/morse_smale_segmentation.png")
