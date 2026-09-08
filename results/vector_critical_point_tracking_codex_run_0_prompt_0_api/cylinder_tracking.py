from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    vector = LoadVectorField(
        load_path=load_path,
        u_array_name="u",
        v_array_name="v",
    )
    critical_points = ComputeVectorCriticalPoints(parent_node_id=vector.id)
    tracked_points = TrackVectorCriticalPoints(
        parent_node_id=critical_points.id,
        time_steps=3,
        backend="partial",
    )

    field_view = EmbedVectorField(parent_node_id=vector.id)
    tracked_points_view = EmbedTrackedVectorFieldCriticalPoints(
        parent_node_id=tracked_points.id,
        radius=4.0,
    )
    return [field_view, tracked_points_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1200, 800]
    view.OrientationAxesVisibility = 0
    view.CenterOfRotation = [74.5, 224.5, 0.0]
    view.CameraPosition = [74.5, 224.5, 1000.0]
    view.CameraFocalPoint = [74.5, 224.5, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = 250.0
    view.CameraParallelProjection = True
    view.InteractionMode = "2D"
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            raise ValueError("-i must be a .vti file or time-series directory")
        time_steps = len(os.listdir(args.i))
        basename = os.path.basename(args.i)
        dirname = os.path.dirname(args.i)
        for i in range(1, time_steps + 1):
            expected = f"{dirname}/{basename}/{basename}{i}.vti"
            if not os.path.isfile(expected):
                raise ValueError(f"invalid time series; missing {expected}")
    if args.t > time_steps:
        raise ValueError("time step is too high")
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-t", type=int, default=3)
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
    for node in nodes_to_visualize:
        node.visualize(args.t, render_view)
    SaveScreenshot(args.o, render_view)
