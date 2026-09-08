from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

wd = os.path.dirname(os.path.abspath(__file__))

# --- Load the tensor field ---
brain = XMLImageDataReader(FileName=[os.path.join(wd, "brain.vti")])
brain.PointArrayStatus = ['A', 'B', 'D']

renderView1 = GetActiveViewOrCreate('RenderView')

brainDisplay = Show(brain, renderView1, 'UniformGridRepresentation')
ColorBy(brainDisplay, ('POINTS', 'A'))
brainDisplay.SetScalarBarVisibility(renderView1, True)
brainDisplay.RescaleTransferFunctionToDataRange(True)

# --- Load the degenerate points ---
degpts = XMLPolyDataReader(FileName=[os.path.join(wd, "degenerate_points.vtp")])

# Split into trisectors (Type == 0) and wedges (Type == 1)
trisectors = Threshold(Input=degpts)
trisectors.Scalars = ['POINTS', 'Type']
trisectors.LowerThreshold = 0
trisectors.UpperThreshold = 0

wedges = Threshold(Input=degpts)
wedges.Scalars = ['POINTS', 'Type']
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1

# --- Glyph trisectors as pink spheres, radius 1 ---
triGlyph = Glyph(Input=trisectors, GlyphType='Sphere')
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleArray = ['POINTS', 'No scale array']
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = 'All Points'

triDisplay = Show(triGlyph, renderView1)
triDisplay.DiffuseColor = [1.0, 0.4117647058823529, 0.7058823529411765]  # pink
ColorBy(triDisplay, None)

# --- Glyph wedges as white spheres, radius 1 ---
wedGlyph = Glyph(Input=wedges, GlyphType='Sphere')
wedGlyph.GlyphType.Radius = 1.0
wedGlyph.ScaleArray = ['POINTS', 'No scale array']
wedGlyph.ScaleFactor = 1.0
wedGlyph.GlyphMode = 'All Points'

wedDisplay = Show(wedGlyph, renderView1)
wedDisplay.DiffuseColor = [1.0, 1.0, 1.0]  # white
ColorBy(wedDisplay, None)

renderView1.ResetCamera()
renderView1.Background = [0.2, 0.2, 0.2]
renderView1.OrientationAxesVisibility = 0

Render(renderView1)
SaveScreenshot(os.path.join(wd, "degenerate_points_render.png"), renderView1, ImageResolution=[1400, 900])

print("Saved degenerate_points_render.png")
