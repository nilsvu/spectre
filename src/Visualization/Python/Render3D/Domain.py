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


@click.command(name="domain")
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
@click.option("zoom_factor", "--zoom", help="Zoom factor.", default=1.0)
def render_domain_command(
    xmf_file,
    output,
    time_step,
    animate,
    zoom_factor,
):
    """Renders the domain geometry"""
    import paraview.simple as pv

    # Load data
    volume_data = pv.XDMFReader(
        registrationName="VolumeData", FileNames=[xmf_file]
    )

    render_view = pv.GetActiveViewOrCreate("RenderView")
    render_view.AxesGrid.Visibility = 1

    # Slice the domain
    slice = pv.Slice(registrationName="Slice", Input=volume_data)
    slice.SliceType = "Plane"
    slice.SliceType.Origin = clip_origin
    slice.SliceType.Normal = clip_normal

    # Show slice
    slice_display = pv.Show(slice, render_view)
    slice_display.Representation = "Surface With Edges"
    slice_display.EdgeColor = 3 * [0.0]
    pv.ColorBy(slice_display, None)

    # Show time annotation
    annotate_time = pv.AnnotateTimeFilter(
        registrationName="AnnotateTimeFilter", Input=volume_data
    )
    pv.Show(annotate_time, render_view)

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
    render_domain_command(help_option_names=["-h", "--help"])
