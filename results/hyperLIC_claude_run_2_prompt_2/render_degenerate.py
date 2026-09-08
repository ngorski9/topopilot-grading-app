from paraview.simple import *

# --- tensor field background ---
reader = XMLImageDataReader(FileName=['/workspace/brain.vti'])
reader.UpdatePipeline()

calc = Calculator(Input=reader)
calc.ResultArrayName = 'anisotropy'
calc.Function = 'sqrt((A-D)^2+4*B^2)'
calc.UpdatePipeline()

view = CreateRenderView()
view.ViewSize = [1200, 800]
view.OrientationAxesVisibility = 0
view.Background = [0, 0, 0]
view.InteractionMode = '2D'

fieldDisp = Show(calc, view)
fieldDisp.Representation = 'Surface'
ColorBy(fieldDisp, ('POINTS', 'anisotropy'))
fieldLUT = GetColorTransferFunction('anisotropy')
fieldLUT.ApplyPreset('Grayscale', True)
fieldDisp.SetScalarBarVisibility(view, False)
fieldDisp.RescaleTransferFunctionToDataRange(True)

def add_points(csv_file, color, radius=1.0):
    csv = CSVReader(FileName=[csv_file])
    csv.UpdatePipeline()
    pts = TableToPoints(Input=csv)
    pts.XColumn = 'x'
    pts.YColumn = 'y'
    pts.ZColumn = 'z'
    pts.UpdatePipeline()

    sphere = Sphere()
    sphere.Radius = radius
    sphere.ThetaResolution = 16
    sphere.PhiResolution = 16

    glyph = Glyph(Input=pts, GlyphType=sphere)
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = 'All Points'
    glyph.UpdatePipeline()

    disp = Show(glyph, view)
    disp.Representation = 'Surface'
    disp.DiffuseColor = color
    disp.AmbientColor = color
    return glyph, disp

# trisectors -> pink, wedges -> white
add_points('/workspace/trisector_points.csv', [1.0, 0.4, 0.7])
add_points('/workspace/wedge_points.csv', [1.0, 1.0, 1.0])

ResetCamera(view)
cam = view.GetActiveCamera()
cam.SetParallelProjection(1)
Render(view)
SaveScreenshot('/workspace/brain_degenerate_points.png', view, ImageResolution=[1600, 1000])
print('saved screenshot')
