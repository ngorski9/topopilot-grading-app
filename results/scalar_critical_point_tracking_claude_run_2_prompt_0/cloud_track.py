import paraview.simple as pv

pv.LoadDistributedPlugin('TopologyToolKit', ns=globals())

files = ['/workspace/cloud/cloud1.vti', '/workspace/cloud/cloud2.vti', '/workspace/cloud/cloud3.vti']
print('Using files:', files)

# --- 0) Time-varying reader for visualizing the original scalar field ---
reader = pv.XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

pds = []
for f in files:
    r = pv.XMLImageDataReader(FileName=[f])
    r.PointArrayStatus = ['Scalars_']

    # normalize to [0,1] -- also de-duplicates ties in the (integer-valued)
    # scalar field, which otherwise creates thousands of degenerate/noise
    # persistence pairs and makes the Wasserstein assignment intractable
    norm = TTKScalarFieldNormalizer(Input=r)
    norm.ScalarField = ['POINTS', 'Scalars_']

    # --- persistence simplification, threshold 0.5 ---
    simplify = TTKTopologicalSimplificationByPersistence(Input=norm)
    simplify.InputArray = ['POINTS', 'Scalars_']
    simplify.PersistenceThreshold = 0.5
    simplify.ThresholdIsAbsolute = 1
    simplify.PairType = 'Extremum-Saddle'

    # persistence diagram of the simplified (piecewise-linear) field
    pd = TTKPersistenceDiagram(Input=simplify)
    pd.ScalarField = ['POINTS', 'Scalars_']
    pd.UpdatePipeline()
    pds.append(pd)

# --- track PL maxima across the 3 time steps using earth mover's distance
#     (Wasserstein optimal-transport assignment) ---
tracking = TTKTrackingFromPersistenceDiagrams(Input=pds)
tracking.Assignmentmethod = 'ttk: pMunkres (Wasserstein), Gabow-Tarjan (Bottleneck)'
tracking.pparameter = '2'
tracking.Persistencethreshold = 0.5
tracking.UpdatePipeline()

nCells = tracking.GetDataInformation().GetNumberOfCells()
nPoints = tracking.GetDataInformation().GetNumberOfPoints()
print(f'Tracked trajectory segments: {nCells}, tracked point samples: {nPoints}')

# ==================== Rendering ====================
view = pv.GetActiveViewOrCreate('RenderView')
view.ViewSize = [1200, 900]

# 1) original scalar field, warm-cold colormap
origDisplay = pv.Show(reader, view)
origDisplay.Representation = 'Surface'
pv.ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origLUT = pv.GetColorTransferFunction('Scalars_')
origLUT.ApplyPreset('Warm to Cold', True)
pv.GetScalarBar(origLUT, view).Title = 'Scalars_'
origDisplay.SetScalarBarVisibility(view, True)

# 2) tracked maxima trajectories, rendered as tubes
trackGeom = pv.ExtractSurface(Input=tracking)
tube = pv.Tube(Input=trackGeom)
tube.Radius = 0.5
tubeDisplay = pv.Show(tube, view)
tubeDisplay.DiffuseColor = [0.0, 0.0, 0.0]

# 3) tracked critical points (PL maxima) as spheres, radius 2
glyph = pv.Glyph(Input=tracking, GlyphType='Sphere')
glyph.GlyphType.Radius = 2.0
glyph.ScaleArray = ['POINTS', 'No scale array']
glyph.ScaleFactor = 1.0
glyph.GlyphMode = 'All Points'
glyphDisplay = pv.Show(glyph, view)
pv.ColorBy(glyphDisplay, ('POINTS', 'TimeStep'))
maxLUT = pv.GetColorTransferFunction('TimeStep')
maxLUT.ApplyPreset('Warm to Cold', True)
maxLUT.RescaleTransferFunction(0, 1)
glyphDisplay.SetScalarBarVisibility(view, True)

view.ResetCamera()
view.CameraPosition = [128, 128, 500]
view.CameraFocalPoint = [128, 128, 0]
pv.Render(view)
pv.SaveScreenshot('/workspace/cloud_tracking.png', view, ImageResolution=[1200, 900])
print('Saved screenshot to /workspace/cloud_tracking.png')
