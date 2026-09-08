from paraview.simple import *
import paraview

paraview.simple._DisableFirstRenderCameraReset()

LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())

# --- load the tensor field (A, B, D plus a precomputed 3x3 "Tensor" array) ---
brain = XMLImageDataReader(FileName=['/workspace/brain_tensor.vti'])
brain.PointArrayStatus = ['A', 'B', 'D', 'Tensor']

# TTK tensor glyphs to visualize the field
tensorGlyph = TensorGlyph(Input=brain)
tensorGlyph.Tensors = ['POINTS', 'Tensor']
tensorGlyph.ScaleFactor = 1.0

view = GetActiveViewOrCreate('RenderView')

glyphDisplay = Show(tensorGlyph, view)
ColorBy(glyphDisplay, ('POINTS', 'A'))
glyphDisplay.SetScalarBarVisibility(view, True)

# also show the underlying grid faintly for context
brainDisplay = Show(brain, view)
brainDisplay.Representation = 'Surface'
ColorBy(brainDisplay, ('POINTS', 'A'))
brainDisplay.SetScalarBarVisibility(view, False)
brainDisplay.Opacity = 0.3

# --- degenerate points ---
degen = XMLPolyDataReader(FileName=['/workspace/degenerate_points.vtp'])
degen.PointArrayStatus = ['type']

trisectors = Threshold(Input=degen)
trisectors.Scalars = ['POINTS', 'type']
trisectors.LowerThreshold = 0
trisectors.UpperThreshold = 0

wedges = Threshold(Input=degen)
wedges.Scalars = ['POINTS', 'type']
wedges.LowerThreshold = 1
wedges.UpperThreshold = 1

triGlyph = Glyph(Input=trisectors, GlyphType='Sphere')
triGlyph.GlyphType.Radius = 1.0
triGlyph.ScaleFactor = 1.0
triGlyph.ScaleArray = ['POINTS', 'No scale array']
triGlyph.GlyphMode = 'All Points'

wedGlyph = Glyph(Input=wedges, GlyphType='Sphere')
wedGlyph.GlyphType.Radius = 1.0
wedGlyph.ScaleFactor = 1.0
wedGlyph.ScaleArray = ['POINTS', 'No scale array']
wedGlyph.GlyphMode = 'All Points'

triDisplay = Show(triGlyph, view)
triDisplay.Representation = 'Surface'
triDisplay.AmbientColor = [1.0, 0.0, 0.75]
triDisplay.DiffuseColor = [1.0, 0.0, 0.75]

wedDisplay = Show(wedGlyph, view)
wedDisplay.Representation = 'Surface'
wedDisplay.AmbientColor = [1.0, 1.0, 1.0]
wedDisplay.DiffuseColor = [1.0, 1.0, 1.0]

view.ResetCamera()
view.OrientationAxesVisibility = 0
Render(view)

SaveScreenshot('/workspace/degenerate_points.png', view, ImageResolution=[1600, 1000])
SaveState('/workspace/brain_degenerate_points.pvsm')

print('Done. Screenshot saved to /workspace/degenerate_points.png')
print('ParaView state saved to /workspace/brain_degenerate_points.pvsm')
