from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(
        parent_node_id=scalar.id,
        epsilon=0.04,
        threshold_is_absolute=False,
    )
    diagram = ComputePersistenceDiagram(parent_node_id=simplified.id)
    diagram_view = EmbedPersistenceDiagram(parent_node_id=diagram.id)
    return [diagram_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 900]
    view.OrientationAxesVisibility = 0
    view.CameraPosition = [0.0, 0.0, 100.0]
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 1.0
    view.InteractionMode = "2D"
    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a TopoPilot persistence diagram.")
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    args = parser.parse_args()

    config.set_random_seed(DEFAULT_RANDOM_SEED)
    config.change_session_folder(".")
    Path("cache").mkdir(exist_ok=True)

    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(1, render_view)
    render_view.ResetCamera()
    SaveScreenshot(args.o, render_view)
