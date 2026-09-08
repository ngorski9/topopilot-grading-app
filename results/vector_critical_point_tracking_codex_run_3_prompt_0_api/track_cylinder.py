from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    """Track cylinder vector-field critical points over its first three frames."""
    vector = LoadVectorField(load_path, u_array_name="u", v_array_name="v")
    critical_points = ComputeVectorCriticalPoints(parent_node_id=vector.id)
    tracked_points = TrackVectorCriticalPoints(
        parent_node_id=critical_points.id,
        time_steps=3,
        backend="partial",
    )

    vector_view = EmbedVectorField(parent_node_id=vector.id)
    tracked_points_view = EmbedTrackedVectorFieldCriticalPoints(
        parent_node_id=tracked_points.id,
        radius=4.0,
    )

    return [vector_view, tracked_points_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [900, 1200]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [74.5, 224.5, 0.0]
    view.CameraPosition = [74.5, 224.5, 1000.0]
    view.CameraFocalPoint = [74.5, 224.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = 245.0
    view.CameraParallelProjection = True
    view.InteractionMode = "2D"
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            raise ValueError("-i must be a .vti file or a time-series directory")
        time_steps = len(os.listdir(args.i))
        basename, dirname = os.path.basename(args.i), os.path.dirname(args.i)
        for i in range(1, time_steps + 1):
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{i}.vti"):
                raise ValueError("invalid time-series directory")
    if args.t > time_steps:
        raise ValueError("time step is too high")
    if not args.o or args.o[-4:] != ".png":
        raise ValueError("this script renders a selected time step to a PNG")
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-t", type=int, required=True)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    validate_arguments(args)
    config.set_random_seed(args.seed)
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
