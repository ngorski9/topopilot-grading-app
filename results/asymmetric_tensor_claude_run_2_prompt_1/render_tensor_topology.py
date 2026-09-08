from paraview.simple import *

# ---- eigenvector partition (10 px / unit square resampled field) ----
part = XMLImageDataReader(FileName=['/workspace/eigenvector_partition.vti'])
part.UpdatePipeline()

renderView = CreateView('RenderView')
renderView.ViewSize = [1000, 1000]
renderView.OrientationAxesVisibility = 0
renderView.Background = [1, 1, 1]

partDisplay = Show(part, renderView, 'UniformGridRepresentation')
ColorBy(partDisplay, ('POINTS', 'EigenvectorPartition'))
partLUT = GetColorTransferFunction('EigenvectorPartition')
partLUT.RGBPoints = [0.0, 0.55, 0.55, 0.55,   # complex domain -> gray
                      1.0, 0.85, 0.85, 0.85]  # real domain -> light gray
partLUT.ColorSpace = 'RGB'
partDisplay.SetScalarBarVisibility(renderView, False)
partDisplay.SetRepresentationType('Surface')

# ---- degenerate points ----
wedges = XMLPolyDataReader(FileName=['/workspace/wedges.vtp'])
tris = XMLPolyDataReader(FileName=['/workspace/trisectors.vtp'])

sphere = Sphere()
sphere.Radius = 1.0
sphere.ThetaResolution = 16
sphere.PhiResolution = 16

wedgeGlyph = Glyph(Input=wedges, GlyphType=sphere)
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = 'All Points'
wedgeGlyph.GlyphType.Radius = 1.0

triGlyph = Glyph(Input=tris, GlyphType=Sphere())
triGlyph.ScaleFactor = 1.0
triGlyph.GlyphMode = 'All Points'
triGlyph.GlyphType.Radius = 1.0

wedgeDisplay = Show(wedgeGlyph, renderView)
wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]   # white wedges
wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]

triDisplay = Show(triGlyph, renderView)
triDisplay.DiffuseColor = [1.0, 0.4118, 0.7059]  # pink trisectors
triDisplay.AmbientColor = [1.0, 0.4118, 0.7059]

renderView.ResetCamera()
renderView.CameraParallelProjection = 1
renderView.Update()

Render(renderView)
SaveScreenshot('/workspace/tensor_field_topology.png', renderView, ImageResolution=[1200, 1200])
print('saved screenshot')
