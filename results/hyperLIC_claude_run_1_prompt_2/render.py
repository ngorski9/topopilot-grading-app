from paraview.simple import *
from paraview import servermanager as sm

paraview.simple._DisableFirstRenderCameraReset()

# --- load tensor field ---
brain = XMLImageDataReader(FileName=['/workspace/brain.vti'])
brain.PointArrayStatus = ['A', 'B', 'D']

calc = Calculator(Input=brain)
calc.ResultArrayName = 'Anisotropy'
calc.Function = 'sqrt((A-D)^2+4*B^2)'

renderView = GetActiveViewOrCreate('RenderView')
renderView.OrientationAxesVisibility = 0
renderView.Background = [1.0, 1.0, 1.0]

fieldDisp = Show(calc, renderView)
fieldDisp.Representation = 'Surface'
ColorBy(fieldDisp, ('POINTS', 'Anisotropy'))
fieldDisp.RescaleTransferFunctionToDataRange(True)
aniLUT = GetColorTransferFunction('Anisotropy')
aniLUT.ApplyPreset('Cool to Warm', True)
fieldDisp.SetScalarBarVisibility(renderView, True)

# --- load degenerate points ---
deg = XMLPolyDataReader(FileName=['/workspace/degenerate_points.vtp'])

tris = Threshold(Input=deg)
tris.Scalars = ['POINTS', 'type']
tris.LowerThreshold = 0
tris.UpperThreshold = 0

wedges = Threshold(Input=deg)
wedges.Scalars = ['POINTS', 'type']
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1

triGlyph = Glyph(Input=tris, GlyphType='Sphere')
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = 'All Points'

wedGlyph = Glyph(Input=wedges, GlyphType='Sphere')
wedGlyph.GlyphType.Radius = 1.0
wedGlyph.ScaleFactor = 1.0
wedGlyph.GlyphMode = 'All Points'

triDisp = Show(triGlyph, renderView)
triDisp.Representation = 'Surface'
ColorBy(triDisp, None)
triDisp.AmbientColor = [1.0, 0.4117647058823529, 0.7058823529411765]
triDisp.DiffuseColor = [1.0, 0.4117647058823529, 0.7058823529411765]

wedDisp = Show(wedGlyph, renderView)
wedDisp.Representation = 'Surface'
ColorBy(wedDisp, None)
wedDisp.AmbientColor = [1.0, 1.0, 1.0]
wedDisp.DiffuseColor = [1.0, 1.0, 1.0]
wedDisp.EdgeColor = [0.0, 0.0, 0.0]

renderView.ResetCamera()
renderView.CameraPosition = [53.5, 32.5, 250]
renderView.CameraFocalPoint = [53.5, 32.5, 0]
renderView.CameraViewUp = [0, 1, 0]
renderView.ResetCamera()

renderView.ViewSize = [1400, 900]
Render(renderView)
SaveScreenshot('/workspace/brain_degenerate_points.png', renderView, ImageResolution=[1400, 900])
print("Saved /workspace/brain_degenerate_points.png")
