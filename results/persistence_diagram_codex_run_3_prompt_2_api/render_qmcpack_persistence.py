from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path


def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(
        parent_node_id=scalar.id,
        epsilon=0.04,
        threshold_is_absolute=False,
    )
    diagram = ComputePersistenceDiagram(parent_node_id=simplified.id)
    diagram_view = EmbedPersistenceDiagram(
        parent_node_id=diagram.id,
        ball_radius=0.012,
        tube_radius=0.006,
    )
    return [diagram_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1200, 1000]
    view.OrientationAxesVisibility = 0
    view.CameraPosition = [0.0, 0.0, 10.0]
    view.CameraFocalPoint = [0.0, 0.0, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 1.25
    view.InteractionMode = "2D"
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            raise ValueError("-i must name a .vti file or a time-series directory")
        time_steps = len(os.listdir(args.i))
    if args.t > time_steps:
        raise ValueError("time step is too high")
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-q", default=".")
    args = parser.parse_args()

    validate_arguments(args)
    config.set_random_seed(0)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)

    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        node.visualize(args.t, render_view)
    SaveScreenshot(args.o, render_view)
