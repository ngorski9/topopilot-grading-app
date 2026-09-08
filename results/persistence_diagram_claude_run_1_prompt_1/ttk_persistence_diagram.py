from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

LoadDistributedPlugin('TopologyToolKit', ns=globals())

# --- Load the scalar field ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

# --- Compute the piecewise-linear persistence diagram (sublevel set filtration) ---
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ['POINTS', 'Scalars_']
pd.InputOffsetField = ['POINTS', 'Scalars_']
pd.Backend = 'FTM (IEEE TPSD 2019)'   # exact PL persistence diagram computation
pd.UpdatePipeline()

# --- Persistence simplification: keep only pairs with persistence >= 0.04 ---
simplified = Threshold(Input=pd)
simplified.Scalars = ['CELLS', 'Persistence']
simplified.LowerThreshold = 0.04
simplified.UpperThreshold = 1e12
simplified.ThresholdMethod = 'Between'
simplified.UpdatePipeline()

# --- Compute Death = Birth + Persistence for plotting ---
withDeath = Calculator(Input=simplified)
withDeath.AttributeType = 'Cell Data'
withDeath.ResultArrayName = 'Death'
withDeath.Function = 'Birth + Persistence'
withDeath.UpdatePipeline()

# --- Keep only order-0 (minimum-saddle) and order-2 (saddle-maximum) pairs ---
order0 = Threshold(Input=withDeath)
order0.Scalars = ['CELLS', 'PairType']
order0.LowerThreshold = 0
order0.UpperThreshold = 0
order0.ThresholdMethod = 'Between'
order0.UpdatePipeline()

order2 = Threshold(Input=withDeath)
order2.Scalars = ['CELLS', 'PairType']
order2.LowerThreshold = 2
order2.UpperThreshold = 2
order2.ThresholdMethod = 'Between'
order2.UpdatePipeline()

merged = AppendDatasets(Input=[order0, order2])
merged.UpdatePipeline()

print('Order-0 pairs (min-saddle):', order0.GetDataInformation().GetNumberOfCells())
print('Order-2 pairs (saddle-max):', order2.GetDataInformation().GetNumberOfCells())
print('Total simplified pairs (all dims, persistence>=0.04):', simplified.GetDataInformation().GetNumberOfCells())

# --- Display the persistence diagram (birth/death scatter plot) ---
view = CreateView('XYChartView')
view.ViewSize = [1000, 800]

disp0 = Show(order0, view)
disp0.AttributeType = 'Cell Data'
disp0.XArrayName = 'Birth'
disp0.SeriesVisibility = ['Death']
disp0.SeriesLineStyle = ['Death', '0']
disp0.SeriesMarkerStyle = ['Death', '5']
disp0.SeriesLabel = ['Death', 'Order 0 (min-saddle)']
disp0.SeriesColor = ['Death', '0.12', '0.47', '0.71']

disp2 = Show(order2, view)
disp2.AttributeType = 'Cell Data'
disp2.XArrayName = 'Birth'
disp2.SeriesVisibility = ['Death']
disp2.SeriesLineStyle = ['Death', '0']
disp2.SeriesMarkerStyle = ['Death', '4']
disp2.SeriesLabel = ['Death', 'Order 2 (saddle-max)']
disp2.SeriesColor = ['Death', '0.84', '0.15', '0.16']

view.LeftAxisTitle = 'Death'
view.BottomAxisTitle = 'Birth'
view.ChartTitle = 'Persistence Diagram (orders 0 & 2, sublevel set, persistence >= 0.04)'

Render(view)
SaveScreenshot('/workspace/persistence_diagram.png', view, ImageResolution=[1000, 800])

print('Persistence diagram saved to /workspace/persistence_diagram.png')
