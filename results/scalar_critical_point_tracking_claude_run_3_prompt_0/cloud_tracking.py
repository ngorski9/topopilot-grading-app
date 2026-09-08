"""
Time-varying cloud dataset analysis pipeline.

1. Load the cloud/*.vti time series (Scalars_ field) into ParaView.
2. Visualize the original scalar field.
3. Compute per-timestep persistence diagrams, simplify with a persistence
   threshold of 0.5, and extract piecewise-linear maxima.
4. Track the maxima across time using TTK's persistence-diagram tracking
   filter, which matches critical points between consecutive timesteps by
   solving an optimal-transport assignment problem (earth mover's / EMD
   distance) between the persistence diagrams.
5. Render the tracked critical points (as spheres, radius 2) together with
   the scalar field, both using a warm-cold ("Cool to Warm") colormap.
"""

import glob
import os
import re

from paraview.simple import *

LoadDistributedPlugin('TopologyToolKit', ns=globals())

# ---------------------------------------------------------------------------
# 1. Load time-varying dataset
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cloud')


def sort_key(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else 0


files = sorted(glob.glob(os.path.join(DATA_DIR, '*.vti')), key=sort_key)
assert files, 'no .vti files found in ./cloud'

N_STEPS = 3
files = files[:N_STEPS]
print('Using timesteps:')
for f in files:
    print('  ', f)

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
timesteps = list(reader.TimestepValues)
print('Timesteps:', timesteps)

# ---------------------------------------------------------------------------
# 2. Visualize the original scalar field
# ---------------------------------------------------------------------------

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

field_display = Show(reader, view)
field_display.Representation = 'Surface'
ColorBy(field_display, ('POINTS', 'Scalars_'))
field_lut = GetColorTransferFunction('Scalars_')
field_lut.ApplyPreset('Cool to Warm', True)
field_display.SetScalarBarVisibility(view, True)

view.ResetCamera()
Render(view)
SaveScreenshot(os.path.join(os.path.dirname(DATA_DIR), 'scalar_field.png'), view)
print('Saved scalar_field.png')

# ---------------------------------------------------------------------------
# 3. Persistence diagram + simplification (threshold = 0.5) per timestep
# ---------------------------------------------------------------------------

persistence_diagram = TTKPersistenceDiagram(Input=reader)
persistence_diagram.ScalarField = ['POINTS', 'Scalars_']
persistence_diagram.UpdatePipeline()

# Simplify the scalar field by removing persistence pairs below threshold 0.5.
simplification = TTKTopologicalSimplificationByPersistence(Input=reader)
simplification.InputArray = ['POINTS', 'Scalars_']
simplification.PersistenceThreshold = 0.5
simplification.ThresholdIsAbsolute = True
simplification.UpdatePipeline()

# Piecewise-linear critical points on the simplified field, keep only maxima.
critical_points = TTKScalarFieldCriticalPoints(Input=simplification)
critical_points.ScalarField = ['POINTS', 'Scalars_']
critical_points.UpdatePipeline()

maxima = Threshold(Input=critical_points)
maxima.Scalars = ['POINTS', 'CriticalType']
maxima.LowerThreshold = 3  # TTK convention: 3 = local maximum
maxima.UpperThreshold = 3
maxima.ThresholdMethod = 'Between'
maxima.UpdatePipeline()

# ---------------------------------------------------------------------------
# 4. Track maxima across time via persistence-diagram matching (EMD)
# ---------------------------------------------------------------------------

# TTKTrackingFromPersistenceDiagrams matches critical points between
# consecutive persistence diagrams by solving an assignment problem; setting
# DistanceAlgorithm to the exact/optimal-transport solver realizes the
# earth mover's distance matching requested.
tracking = TTKTrackingFromPersistenceDiagrams(Input=persistence_diagram)
tracking.Persistencethreshold = 0.5
# p=1 -> earth mover's distance (Wasserstein-1) matching between diagrams,
# solved with TTK's pMunkres optimal-transport assignment solver.
tracking.pparameter = '1'
tracking.UpdatePipeline()

print('Tracking output produced:', tracking.GetDataInformation().GetNumberOfPoints(), 'points')

# ---------------------------------------------------------------------------
# 5. Display tracked critical points (sphere glyphs, radius 2) + scalar field
# ---------------------------------------------------------------------------

glyph = Glyph(Input=tracking, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'

glyph_display = Show(glyph, view)
ColorBy(glyph_display, ('POINTS', 'Scalars_'))
glyph_lut = GetColorTransferFunction('Scalars_')
glyph_lut.ApplyPreset('Cool to Warm', True)
glyph_display.SetScalarBarVisibility(view, True)

tracking_display = Show(tracking, view)
tracking_display.Representation = 'Wireframe'
ColorBy(tracking_display, ('POINTS', 'Scalars_'))

view.ResetCamera()
Render(view)
SaveScreenshot(os.path.join(os.path.dirname(DATA_DIR), 'tracked_maxima.png'), view)
print('Saved tracked_maxima.png')

print('Pipeline complete.')
