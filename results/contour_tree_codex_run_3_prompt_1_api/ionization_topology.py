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
    original_field = EmbedScalarField(parent_node_id=scalar.id, colormap="viridis")

    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.1)
    contour_tree = ComputeContourTree(parent_node_id=simplified.id)
    contour_tree_view = EmbedContourTree(
        parent_node_id=contour_tree.id,
        ball_radius=2.0,
        tube_radius=1.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [original_field, contour_tree_view], {original_field.id: ([0.85, 0.15], 0.70, "Ionization")}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [149.5, 61.5, 61.5]
    view.CameraPosition = [520.0, 410.0, 510.0]
    view.CameraFocalPoint = [149.5, 61.5, 61.5]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = False
    view.CameraViewAngle = 30.0
    view.InteractionMode = "3D"
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            raise SystemExit("input must be a .vti file or a time-series directory")
        time_steps = len(os.listdir(args.i))
        basename, dirname = os.path.basename(args.i), os.path.dirname(args.i)
        for index in range(1, time_steps + 1):
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{index}.vti"):
                raise SystemExit(f"invalid time series: missing {basename}{index}.vti")
    if args.t > time_steps:
        raise SystemExit("time step is too high")
    if args.x and args.t == -1 and time_steps > 1:
        raise SystemExit("cannot run interactive mode on multiple time steps")
    if args.o and args.x:
        raise SystemExit("cannot specify output with interactive mode")
    if not args.o and not args.x:
        raise SystemExit("specify output or interactive mode")
    if args.t == -1 and time_steps > 1 and not args.o.endswith(".mp4"):
        raise SystemExit("all time steps require .mp4 output")
    if args.t != -1 and not args.x and not args.o.endswith(".png"):
        raise SystemExit("one time step requires .png output")
    return 1 if args.t == -1 and time_steps == 1 else time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-x", action=argparse.BooleanOptionalAction)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    time_steps = validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)

    def render_step(time_step, output_path):
        render_view = create_view()
        for node in nodes_to_visualize:
            props = node.visualize(time_step, render_view)
            if node.id in color_bar_info:
                position, length, title = color_bar_info[node.id]
                bar = GetScalarBar(props.LookupTable, render_view)
                bar.WindowLocation, bar.Position, bar.ScalarBarLength = "Any Location", position, length
                bar.Title, bar.ComponentTitle = title, ""
                props.SetScalarBarVisibility(render_view, True)
        SaveScreenshot(output_path, render_view)

    if args.t != -1:
        render_step(args.t, args.o)
    else:
        for time_step in range(1, time_steps + 1):
            render_step(time_step, f"{args.q}/frame{time_step:08d}.png")
            Disconnect()
            tree.clear_evaluation_cache()
            Connect()
        os.system(f"ffmpeg -y -framerate 4 -i {args.q}/frame%08d.png -c:v libx264 -pix_fmt yuv420p {args.o}")
        for time_step in range(1, time_steps + 1):
            os.remove(f"{args.q}/frame{time_step:08d}.png")
