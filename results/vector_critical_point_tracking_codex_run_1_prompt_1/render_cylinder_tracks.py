#!/usr/bin/env /opt/conda/bin/pvpython
from paraview.simple import *
from pathlib import Path

root = Path('/workspace')
field = PVDReader(registrationName='Cylinder vector field (steps 1, 11, 21)', FileName=str(root/'cylinder_3steps.pvd'))
field.PointArrays = ['u', 'v']
calc = Calculator(registrationName='Vector field: (u, v, 0)', Input=field)
calc.ResultArrayName = 'Velocity'; calc.Function = 'u*iHat + v*jHat'
mask = MaskPoints(registrationName='Vector sampling', Input=calc)
mask.OnRatio = 12
glyph = Glyph(registrationName='Original vector field', Input=mask, GlyphType='Arrow')
glyph.OrientationArray = ['POINTS', 'Velocity']; glyph.ScaleArray = ['POINTS', 'Velocity']
glyph.ScaleFactor = 28.0; glyph.GlyphMode = 'All Points'
points = XMLPolyDataReader(registrationName='Tracked critical points', FileName=[str(root/'tracked_critical_points.vtp')])
tracks = XMLPolyDataReader(registrationName='Partial optimal transport tracks', FileName=[str(root/'partial_ot_tracks.vtp')])

view = CreateView('RenderView'); view.ViewSize = [1300, 720]
view.InteractionMode = '2D'; view.OrientationAxesVisibility = 0
view.Background = [0.08, 0.10, 0.14]
gdisp = Show(glyph, view); gdisp.Representation='Surface'; gdisp.DiffuseColor=[0.72,0.78,0.86]; gdisp.Opacity=0.80
pdisp = Show(points, view); pdisp.Representation='Points'; pdisp.PointSize=11.0
ColorBy(pdisp, ('POINTS','TimeStep')); pdisp.RescaleTransferFunctionToDataRange(True, False)
tdisp = Show(tracks, view); tdisp.Representation='Surface'; tdisp.LineWidth=6.0; tdisp.DiffuseColor=[1.0,0.84,0.18]
field.UpdatePipeline(); view.ResetCamera()
view.CameraParallelProjection = 1
view.CameraParallelScale = 250
Render()
SaveScreenshot(str(root/'cylinder_partial_ot_tracks.png'), view, ImageResolution=[1300,720])
SaveState(str(root/'cylinder_partial_ot_tracks.pvsm'))
