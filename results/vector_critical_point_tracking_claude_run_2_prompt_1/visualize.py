"""
Render the cylinder vector field (timestep 2, the middle of the 3 tracked
steps) together with the critical points / trajectories obtained via
partial-optimal-transport tracking, and save a ParaView state (.pvsm) plus
a PNG snapshot.
"""
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# --- vector field (middle timestep) ---
field = XMLImageDataReader(FileName=['cylinder/cylinder2.vti'])
field.PointArrayStatus = ['u', 'v']

calc = Calculator(Input=field)
calc.ResultArrayName = 'velocity'
calc.Function = 'u*iHat+v*jHat'

magnitude = Calculator(Input=calc)
magnitude.ResultArrayName = 'magnitude'
magnitude.Function = 'mag(velocity)'

fieldDisplay = Show(magnitude, GetActiveView() or CreateRenderView())
fieldDisplay.Representation = 'Surface'
ColorBy(fieldDisplay, ('POINTS', 'magnitude'))
fieldDisplay.RescaleTransferFunctionToDataRange(True)
fieldDisplay.SetScalarBarVisibility(GetActiveView(), True)

glyph = Glyph(Input=calc, GlyphType='2D Glyph')
glyph.OrientationArray = ['POINTS', 'velocity']
glyph.ScaleArray = ['POINTS', 'velocity']
glyph.ScaleFactor = 6.0
glyph.GlyphMode = 'Every Nth Point'
glyph.Stride = 40
glyphDisplay = Show(glyph, GetActiveView())
glyphDisplay.AmbientColor = [0.0, 0.0, 0.0]
glyphDisplay.DiffuseColor = [0.0, 0.0, 0.0]

# --- tracked critical points ---
cps = XMLPolyDataReader(FileName=['critical_points.vtp'])
cpsDisplay = Show(cps, GetActiveView())
cpsDisplay.Representation = 'Points'
cpsDisplay.PointSize = 8
ColorBy(cpsDisplay, ('POINTS', 'CriticalType'))
cpsLUT = GetColorTransferFunction('CriticalType')
cpsLUT.RGBPoints = [0.0, 0.85, 0.1, 0.1,   # saddle -> red
                     1.0, 0.1, 0.6, 0.1,   # source -> green
                     2.0, 0.1, 0.1, 0.9]   # sink   -> blue
cpsLUT.InterpretValuesAsCategories = 1
cpsLUT.Annotations = ['0', 'saddle', '1', 'source', '2', 'sink']
cpsLUT.IndexedColors = [0.85, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.9]

# --- trajectories (tracked across the 3 timesteps via partial OT) ---
traj = XMLPolyDataReader(FileName=['trajectories.vtp'])
tube = Tube(Input=traj)
tube.Radius = 0.6
tubeDisplay = Show(tube, GetActiveView())
tubeDisplay.AmbientColor = [1.0, 0.85, 0.0]
tubeDisplay.DiffuseColor = [1.0, 0.85, 0.0]

view = GetActiveView()
view.OrientationAxesVisibility = 0
ResetCamera()
view.CameraPosition = [75, 225, 900]
view.CameraFocalPoint = [75, 225, 0]
Render()

SaveScreenshot('cylinder_tracked_critical_points.png', view, ImageResolution=[1000, 1400])
servermanager.SaveState('cylinder_tracking.pvsm')
print('saved cylinder_tracked_critical_points.png and cylinder_tracking.pvsm')
