# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import rich


def _parse_step(ctx, param, value):
    if value is None:
        return None
    if value.lower() == "first":
        return 0
    if value.lower() == "last":
        return -1
    return int(value)


@click.command(name="clip")
@click.argument(
    "xmf_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    required=True,
    help="Output file. Include extension such as '.png'.",
)
@click.option(
    "--variable",
    "-y",
    help="Variable to plot. Lists available variables when not specified.",
)
@click.option(
    "--time-step",
    "-t",
    callback=_parse_step,
    default="first",
    show_default=True,
    help=(
        "Select a time step. Specify '-1' or 'last' to select the last time"
        " step."
    ),
)
@click.option(
    "--animate", is_flag=True, help="Produce an animation of all time steps."
)
@click.option(
    "log_scale",
    "--log",
    is_flag=True,
    help="Plot variable in log scale.",
)
@click.option(
    "--show-grid",
    is_flag=True,
    help="Show grid lines",
)
@click.option("zoom_factor", "--zoom", help="Zoom factor.", default=1)
@click.option(
    "--clip-origin",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.0),
    help="Origin of the clipping plane",
    show_default=True,
)
@click.option(
    "--clip-normal",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 1.0),
    help="Normal of the clipping plane",
    show_default=True,
)
def render_clip_command(
    xmf_file,
    output,
    variable,
    time_step,
    animate,
    log_scale,
    show_grid,
    zoom_factor,
    clip_origin,
    clip_normal,
):
    """Renders a clip normal to the z-direction.

    XMF_FILE is the path to the XMF file that references the simulation data.
    It is typically generated by the 'generate-xdmf' command.

    This is a quick way to get some insight into the simulation data. For more
    advanced renderings, open the XMF file in an interactive ParaView session,
    or implement rendering commands specialized for your use case.
    """
    import paraview.simple as pv

    # Load data
    volume_data = pv.XDMFReader(
        registrationName="VolumeData", FileNames=[xmf_file]
    )

    # Select variable
    if not variable:
        import rich.columns

        all_variables = volume_data.PointData.keys()
        rich.print(rich.columns.Columns(all_variables))
        return

    render_view = pv.GetActiveViewOrCreate("RenderView")

    # Configure colors
    color_transfer_function = pv.GetColorTransferFunction(variable)
    if log_scale:
        color_transfer_function.UseLogScale = 1

    # Clip
    clip = pv.Clip(registrationName="Clip", Input=volume_data)
    clip.ClipType = "Plane"
    clip.ClipType.Origin = clip_origin
    clip.ClipType.Normal = clip_normal

    # Show clip
    clip_display = pv.Show(clip, render_view)
    if show_grid:
        clip_display.Representation = "Surface With Edges"
    else:
        clip_display.Representation = "Surface"
    pv.ColorBy(clip_display, ("POINTS", variable))
    clip_display.SetScalarBarVisibility(render_view, True)

    # Set resolution
    layout = pv.GetLayout()
    layout.SetSize(1920, 1080)

    # Configure camera
    render_view.ResetCamera(True)
    render_view.CameraViewAngle /= zoom_factor

    if animate:
        pv.SaveAnimation(output, render_view)
    else:
        render_view.ViewTime = volume_data.TimestepValues[time_step]
        pv.SaveScreenshot(output, render_view)


if __name__ == "__main__":
    render_clip_command(help_option_names=["-h", "--help"])
