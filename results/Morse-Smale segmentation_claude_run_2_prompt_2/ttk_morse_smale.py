from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

scalar_field = 'Scalars_'
persistence_threshold = 0.05

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/fracture.vti'])
reader.PointArrayStatus = [scalar_field]

# TTK filters need TetrahedralizeIfNeeded for 3D triangulation; imagedata works directly.

# --- Persistence diagram of the input field ---
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', scalar_field]

# --- Threshold the diagram by persistence to keep pairs above the simplification threshold ---
# Keep everything BELOW the threshold to feed as the simplification "constraint" pairs is not how
# TTKTopologicalSimplification works; instead it takes the ORIGINAL persistence diagram thresholded
# to select which critical point pairs to REMOVE (i.e. keep low persistence pairs so they get simplified away).
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.LowerThreshold = 0.0
threshold.UpperThreshold = persistence_threshold
threshold.ThresholdMethod = 'Between'

# --- Topological simplification using the thresholded (low-persistence) pairs ---
topoSimplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
topoSimplification.Backend = 'Legacy Approach (IEEE VIS 2012)'
topoSimplification.ScalarField = ['POINTS', scalar_field]
topoSimplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# --- Morse-Smale complex on the simplified field ---
morseSmale = TTKMorseSmaleComplex(Input=topoSimplification)
morseSmale.ScalarField = ['POINTS', scalar_field]
morseSmale.CriticalPoints = 1
morseSmale.AscendingSegmentation = 1
morseSmale.DescendingSegmentation = 1
morseSmale.MorseSmaleComplexSegmentation = 1

morseSmale.UpdatePipeline()

# outputs: 0 = Critical Points, 1 = 1-Separatrices, 2 = 2-Separatrices, 3 = Segmentation
segmentation = OutputPort(morseSmale, 3)
criticalPoints = OutputPort(morseSmale, 0)
segmentation.UpdatePipeline()

segManifoldRange = morseSmale.GetClientSideObject().GetOutputDataObject(3).GetPointData().GetArray('MorseSmaleManifold').GetRange()

renderView = GetActiveViewOrCreate('RenderView')

# --- Render segmentation (color by Morse-Smale region id) ---
segDisplay = Show(segmentation, renderView, 'GeometryRepresentation')
segDisplay.ColorArrayName = ['POINTS', 'MorseSmaleManifold']
segColorLUT = GetColorTransferFunction('MorseSmaleManifold')
segColorLUT.ApplyPreset('Rainbow Desaturated', True)
segColorLUT.RescaleTransferFunction(segManifoldRange[0], segManifoldRange[1])
segDisplay.SetScalarBarVisibility(renderView, True)
segDisplay.SetRepresentationType('Surface')

# --- Split critical points by CellDimension and color/size them ---
# Data is 2D (dims 249x121x1): CellDimension 0 = minimum, 1 = saddle, 2 = maximum
minThreshold = Threshold(Input=criticalPoints)
minThreshold.Scalars = ['POINTS', 'CellDimension']
minThreshold.LowerThreshold = 0
minThreshold.UpperThreshold = 0
minThreshold.ThresholdMethod = 'Between'

saddleThreshold = Threshold(Input=criticalPoints)
saddleThreshold.Scalars = ['POINTS', 'CellDimension']
saddleThreshold.LowerThreshold = 1
saddleThreshold.UpperThreshold = 1
saddleThreshold.ThresholdMethod = 'Between'

maxThreshold = Threshold(Input=criticalPoints)
maxThreshold.Scalars = ['POINTS', 'CellDimension']
maxThreshold.LowerThreshold = 2
maxThreshold.UpperThreshold = 2
maxThreshold.ThresholdMethod = 'Between'

def show_points(src, color, radius=2.0):
    glyph = Glyph(Input=src, GlyphType='Sphere')
    glyph.GlyphType.Radius = radius
    glyph.ScaleFactor = 1.0
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.GlyphMode = 'All Points'
    disp = Show(glyph, renderView, 'GeometryRepresentation')
    disp.DiffuseColor = color
    disp.AmbientColor = color
    return glyph, disp

minGlyph, minDisp = show_points(minThreshold, [0.0, 0.0, 1.0], radius=2.0)      # blue minima
saddleGlyph, saddleDisp = show_points(saddleThreshold, [1.0, 1.0, 1.0], radius=2.0)  # white saddles
maxGlyph, maxDisp = show_points(maxThreshold, [1.0, 0.0, 0.0], radius=2.0)      # red maxima

renderView.ResetCamera()
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/fracture_morse_smale.png', renderView, ImageResolution=[1600, 1200])
print('Saved screenshot to /workspace/fracture_morse_smale.png')
