from paraview.simple import *
from pathlib import Path

root = Path(__file__).resolve().parent
brain = XMLImageDataReader(registrationName='brain.vti', FileName=[str(root / 'brain.vti')])
trisectors = XMLPolyDataReader(registrationName='Trisectors (pink)', FileName=[str(root / 'brain_trisectors.vtp')])
wedges = XMLPolyDataReader(registrationName='Wedges (white)', FileName=[str(root / 'brain_wedges.vtp')])

view = CreateView('RenderView')
view.ViewSize = [1200, 800]
view.Background = [0.08, 0.08, 0.11]
view.OrientationAxesVisibility = 0

# Dataset plane: translucent scalar context beneath the singularities.
brainDisplay = Show(brain, view, 'UniformGridRepresentation')
brainDisplay.Representation = 'Surface'
brainDisplay.ColorArrayName = ['POINTS', 'A']
brainDisplay.Opacity = 0.35
brainDisplay.LookupTable = GetColorTransferFunction('A')

def show_spheres(points, name, color):
    sphere = Sphere(registrationName=f'{name} radius 1', Radius=1.0,
                    ThetaResolution=20, PhiResolution=20)
    glyph = Glyph(registrationName=name, Input=points, GlyphType=sphere)
    glyph.OrientationArray = ['POINTS', 'No orientation array']
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 1.0
    display = Show(glyph, view, 'GeometryRepresentation')
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 0.35
    display.Specular = 0.25
    return glyph

show_spheres(trisectors, 'Trisectors — radius 1', [1.0, 0.35, 0.65])
show_spheres(wedges, 'Wedges — radius 1', [1.0, 1.0, 1.0])

view.CameraPosition = [53.5, 32.5, 170.0]
view.CameraFocalPoint = [53.5, 32.5, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 40.0
view.InteractionMode = '2D'
Render(view)
SaveScreenshot(str(root / 'brain_degenerate_points.png'), view, ImageResolution=[1200, 800])
SaveState(str(root / 'brain_degenerate_points.pvsm'))
