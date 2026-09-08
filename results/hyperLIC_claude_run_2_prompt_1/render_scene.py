from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# Load the tensor field
brain = XMLImageDataReader(FileName=['/workspace/brain.vti'])
brain.PointArrayStatus = ['A', 'B', 'D']

# Merge A, B, D into a symmetric tensor for visualization purposes,
# and show trace (A+D) as background scalar field
calc = Calculator(Input=brain)
calc.ResultArrayName = 'trace'
calc.Function = 'A+D'

renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1400, 900]
renderView.OrientationAxesVisibility = 0

fieldDisplay = Show(calc, renderView)
fieldDisplay.Representation = 'Surface'
ColorBy(fieldDisplay, ('POINTS', 'trace'))
fieldDisplay.RescaleTransferFunctionToDataRange(True)
fieldDisplay.SetScalarBarVisibility(renderView, True)
traceLUT = GetColorTransferFunction('trace')
traceLUT.ApplyPreset('Grayscale', True)

# Load degenerate points
deg = XMLPolyDataReader(FileName=['/workspace/degenerate_points.vtp'])

trisectors = Threshold(Input=deg)
trisectors.Scalars = ['POINTS', 'type']
trisectors.LowerThreshold = 0
trisectors.UpperThreshold = 0

wedges = Threshold(Input=deg)
wedges.Scalars = ['POINTS', 'type']
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1

triGlyph = Glyph(Input=trisectors, GlyphType='Sphere')
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleArray = ['POINTS', 'No scale array']
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = 'All Points'

wedGlyph = Glyph(Input=wedges, GlyphType='Sphere')
wedGlyph.GlyphType.Radius = 1.0
wedGlyph.ScaleArray = ['POINTS', 'No scale array']
wedGlyph.ScaleFactor = 1.0
wedGlyph.GlyphMode = 'All Points'

triDisplay = Show(triGlyph, renderView)
triDisplay.Representation = 'Surface'
triDisplay.DiffuseColor = [1.0, 0.4117647058823529, 0.7058823529411765]  # pink
triDisplay.AmbientColor = [1.0, 0.4117647058823529, 0.7058823529411765]

wedDisplay = Show(wedGlyph, renderView)
wedDisplay.Representation = 'Surface'
wedDisplay.DiffuseColor = [1.0, 1.0, 1.0]  # white
wedDisplay.AmbientColor = [1.0, 1.0, 1.0]

renderView.ResetCamera()
renderView.InteractionMode = '2D'
renderView.CameraParallelProjection = 1
renderView.Background = [0.2, 0.2, 0.2]

Render(renderView)
SaveScreenshot('/workspace/degenerate_points_render.png', renderView, ImageResolution=[1400, 900])
print('Saved /workspace/degenerate_points_render.png')
