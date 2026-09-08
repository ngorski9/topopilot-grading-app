from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# --- tensor field dataset ---
reader = XMLImageDataReader(FileName=['/workspace/brain.vti'])
reader.PointArrayStatus = ['A', 'B', 'D']
reader.UpdatePipeline()

calc = Calculator(Input=reader)
calc.ResultArrayName = 'Trace'
calc.Function = 'A+D'
calc.AttributeType = 'Point Data'
calc.UpdatePipeline()

renderView = GetActiveViewOrCreate('RenderView')

fieldDisp = Show(calc, renderView)
ColorBy(fieldDisp, ('POINTS', 'Trace'))
fieldDisp.SetRepresentationType('Surface')
fieldDisp.RescaleTransferFunctionToDataRange(True)
trace_lut = GetColorTransferFunction('Trace')
trace_lut.ApplyPreset('Grayscale', True)

# --- degenerate points, split by type so each gets an unambiguous solid color ---
deg = XMLPolyDataReader(FileName=['/workspace/degenerate_points.vtp'])
deg.UpdatePipeline()

trisectors = Threshold(Input=deg)
trisectors.Scalars = ['POINTS', 'DegenerateType']
trisectors.LowerThreshold = 0
trisectors.UpperThreshold = 0
trisectors.ThresholdMethod = 'Between'
trisectors.UpdatePipeline()

wedges = Threshold(Input=deg)
wedges.Scalars = ['POINTS', 'DegenerateType']
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1
wedges.ThresholdMethod = 'Between'
wedges.UpdatePipeline()

print('trisector points:', trisectors.GetDataInformation().GetNumberOfPoints())
print('wedge points:', wedges.GetDataInformation().GetNumberOfPoints())

triGlyph = Glyph(Input=trisectors, GlyphType='Sphere')
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleArray = ['POINTS', 'No scale array']
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = 'All Points'
triGlyph.GlyphType.ThetaResolution = 16
triGlyph.GlyphType.PhiResolution = 16
triGlyph.UpdatePipeline()

wedgeGlyph = Glyph(Input=wedges, GlyphType='Sphere')
wedgeGlyph.GlyphType.Radius = 1.0
wedgeGlyph.ScaleArray = ['POINTS', 'No scale array']
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = 'All Points'
wedgeGlyph.GlyphType.ThetaResolution = 16
wedgeGlyph.GlyphType.PhiResolution = 16
wedgeGlyph.UpdatePipeline()

triDisp = Show(triGlyph, renderView)
triDisp.SetRepresentationType('Surface')
ColorBy(triDisp, None)
triDisp.AmbientColor = [1.0, 0.4117647, 0.7058823]
triDisp.DiffuseColor = [1.0, 0.4117647, 0.7058823]

wedgeDisp = Show(wedgeGlyph, renderView)
wedgeDisp.SetRepresentationType('Surface')
ColorBy(wedgeDisp, None)
wedgeDisp.AmbientColor = [1.0, 1.0, 1.0]
wedgeDisp.DiffuseColor = [1.0, 1.0, 1.0]

renderView.Background = [0.15, 0.15, 0.15]
renderView.ResetCamera()
renderView.CameraParallelProjection = 1
renderView.Update()
Render(renderView)

SaveScreenshot('/workspace/degenerate_points_view.png', renderView, ImageResolution=[1400, 900])
SaveState('/workspace/brain_degenerate.pvsm')
print('done')
