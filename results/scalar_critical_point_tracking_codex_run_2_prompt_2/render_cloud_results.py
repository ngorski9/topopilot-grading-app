from paraview.simple import *

OUT = '/workspace/output/'
paraview.simple._DisableFirstRenderCameraReset()

original = XMLImageDataReader(registrationName='Original scalar field (t=0)', FileName=[OUT+'cloud_original_t0.vti'])
original.PointArrayStatus = ['Scalars_']
simplified = XMLImageDataReader(registrationName='Persistence simplified (threshold 0.5)', FileName=[OUT+'cloud_persistence_simplified_threshold_0.5.vti'])
simplified.PointArrayStatus = ['Scalars_']
tracks = XMLPolyDataReader(registrationName='Maxima tracks — EMD / Wasserstein-1 (3 time steps)', FileName=[OUT+'cloud_maxima_tracking_emd_3steps.vtp'])

view = CreateView('RenderView')
view.ViewSize = [1200, 850]
view.Background = [0.12, 0.12, 0.16]
view.OrientationAxesVisibility = 0

od = Show(original, view)
od.Representation = 'Surface'
ColorBy(od, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Cool to Warm', True)
lut.RescaleTransferFunction(0.0, 50.0)
od.SetScalarBarVisibility(view, True)

# The simplification is included in the state as a separate, inspectable result.
sd = Show(simplified, view)
sd.Visibility = 0

td = Show(tracks, view)
td.Representation = 'Surface'
td.DiffuseColor = [1.0, 0.86, 0.15]
td.LineWidth = 4.0

# Sphere glyphs mark all tracked maxima at each time step.
glyphs = Glyph(registrationName='Tracked maxima (sphere radius 2)', Input=tracks, GlyphType='Sphere')
glyphs.GlyphType.Radius = 2.0
glyphs.ScaleArray = ['POINTS', 'No scale array']
glyphs.ScaleFactor = 1.0
gd = Show(glyphs, view)
gd.DiffuseColor = [1.0, 0.92, 0.2]

view.CameraPosition = [127.5, 127.5, 380]
view.CameraFocalPoint = [127.5, 127.5, 0]
view.CameraParallelScale = 145
Render(view)
SaveScreenshot(OUT+'cloud_tracking_visualization.png', view, ImageResolution=[1200,850])
SaveState(OUT+'cloud_ttk_emd_tracking.pvsm')
