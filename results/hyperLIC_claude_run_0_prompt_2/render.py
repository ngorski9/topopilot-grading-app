from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

brain = XMLImageDataReader(FileName=["/workspace/brain.vti"])
brain.PointArrayStatus = ['A', 'B', 'D']

calc = Calculator(Input=brain)
calc.ResultArrayName = 'Trace'
calc.Function = 'A+D'

deg = XMLPolyDataReader(FileName=["/workspace/degenerate_points_clean.vtp"])

renderView = GetActiveViewOrCreate('RenderView')
renderView.ViewSize = [1200, 800]
renderView.OrientationAxesVisibility = 0
renderView.Background = [0.15, 0.15, 0.15]

# Show tensor field background colored by trace
dispField = Show(calc, renderView)
dispField.Representation = 'Surface'
ColorBy(dispField, ('POINTS', 'Trace'))
dispField.SetScalarBarVisibility(renderView, True)
traceLUT = GetColorTransferFunction('Trace')
traceLUT.ApplyPreset('Grayscale', True)

# Glyph degenerate points as spheres, radius 1
glyph = Glyph(Input=deg, GlyphType='Sphere')
glyph.GlyphType.Radius = 1.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

dispGlyph = Show(glyph, renderView)
ColorBy(dispGlyph, ('POINTS', 'DegenerateType'))
degLUT = GetColorTransferFunction('DegenerateType')
degLUT.InterpretValuesAsCategories = 1
degLUT.AnnotationsInitialized = 1
degLUT.Annotations = ['0', 'Trisector', '1', 'Wedge']
degLUT.IndexedColors = [1.0, 0.4117647058823529, 0.7058823529411765,   # pink for trisector (0)
                         1.0, 1.0, 1.0]                                 # white for wedge (1)
dispGlyph.SetScalarBarVisibility(renderView, False)

renderView.ResetCamera()
renderView.CameraParallelProjection = 1
renderView.ResetCamera()
renderView.Update()

Render(renderView)
SaveScreenshot("/workspace/brain_degenerate_points.png", renderView, ImageResolution=[1600, 1000])
print("Saved screenshot")
