# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich

import spectre.IO.H5 as spectre_h5

logger = logging.getLogger(__name__)

# // Returns all the observation_ids stored in the volume files. Assumes all
# // volume files have the same observation ids
# std::vector<size_t> get_observation_ids(const std::string& file_prefix,
#                                         const std::string& subfile_name) {
#   const h5::H5File<h5::AccessType::ReadOnly> initial_file(file_prefix + "0.h5",
#                                                           false);
#   const auto& initial_volume_file =
#       initial_file.get<h5::VolumeData>("/" + subfile_name);
#   return initial_volume_file.list_observation_ids();
# }

# // Returns total number of elements for an observation id across all volume data
# // files
# size_t get_number_of_elements(const std::vector<std::string>& input_filenames,
#                               const std::string& subfile_name,
#                               const size_t& observation_id) {
#   size_t total_elements = 0;
#   for (const auto& input_filename : input_filenames) {
#     const h5::H5File<h5::AccessType::ReadOnly> original_file(input_filename,
#                                                              false);
#     const auto& original_volume_file =
#         original_file.get<h5::VolumeData>("/" + subfile_name);
#     total_elements += original_volume_file.get_extents(observation_id).size();
#   }
#   return total_elements;
# }


@click.command(name="combine-h5-vol")
@click.argument(
    "h5files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
)
@click.option(
    "--subfile-name",
    "-d",
    help="subfile name of the volume file in the H5 file",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    help="combined output filename",
)
@click.option(
    "--check-src/--no-check-src",
    default=True,
    show_default=True,
    help=(
        "flag to check src files, True implies src files exist and can be"
        " checked, False implies no src files to check."
    ),
)
def combine_h5_vol_command(h5files, subfile_name, output, check_src):
    """Combines volume data spread over multiple H5 files into a single file

    The typical use case is to combine volume data from multiple nodes into a
    single file, if this is necessary for further processing (e.g. for the
    'extend-connectivity' command). Note that for most use cases it is not
    necessary to combine the volume data into a single file, as most commands
    can operate on multiple input H5 files (e.g. 'generate-xdmf').

    Note that this command does not currently combine volume data from different
    time steps (e.g. from multiple segments of a simulation). All input H5 files
    must contain the same set of observation IDs.
    """
    # CLI scripts should be noops when input is empty
    if not h5files:
        return

    # Print available subfile names and exit
    if not subfile_name:
        spectre_file = spectre_h5.H5File(h5files[0], "r")
        import rich.columns

        rich.print(rich.columns.Columns(spectre_file.all_vol_files()))
        return

    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name
    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]

    if not output.endswith(".h5"):
        output += ".h5"

    logger.debug(f"Processing files: {file_names}")

    #   if (check_src){
    #     if (!h5::check_src_files_match(file_names)) {
    #     ERROR(
    #         "One or more of your files were found to have differing src.tar.gz "
    #         "files, meaning that they may be from differing versions of SpECTRE.");
    #   }
    #   }

    #   if (!h5::check_observation_ids_match(file_names, subfile_name)) {
    #     ERROR(
    #         "One or more of your files were found to have differing observation "
    #         "ids, meaning they may be from different runs of your SpECTRE "
    #         "executable or were corrupted.");
    #   }

    with spectre_h5.H5File(output, "a") as output_file:
        # Instantiates the output file and the .vol subfile to be filled with the
        # combined data later
        logger.debug(f"Creating output file: {output}")
        output_file.insert_vol("/" + subfile_name + ".vol")

    # Obtains list of observation ids to loop over
    observation_ids = get_observation_ids(file_prefix, subfile_name)

    # # Loops over observation ids to write volume data by observation id
    # for obs_id in observation_ids:
    #     # Pre-calculates size of vector to store element data and allocates
    #     # corresponding memory
    #     vector_dim = get_number_of_elements(file_names, subfile_name, obs_id)
    #     std::vector<ElementVolumeData> element_data;
    #     element_data.reserve(vector_dim)

    #     # Loops over input files to append element data into a single vector to be
    #     # stored in a single H5
    #     # TODO: progress bar
    #     for filename in h5files:
    #     original_file = spectre_h5.H5File(filename, "r")
    #     original_volume_file =
    #         original_file.get_vol("/" + subfile_name)
    #     obs_val = original_volume_file.get_observation_value(obs_id)
    #     logger.debug(f"  Processing file: {filename}")

    #     serialized_domain = original_volume_file.get_domain(obs_id)
    #     serialized_functions_of_time =
    #         original_volume_file.get_functions_of_time(obs_id)

    #     # Get vector of element data for this `obs_id` and `file_name`
    #     data_by_element =
    #         std::move(std::get<2>(original_volume_file.get_data_by_element(
    #             obs_val * (1.0 - 4.0 * std::numeric_limits<double>::epsilon()),
    #             obs_val * (1.0 + 4.0 * std::numeric_limits<double>::epsilon()),
    #             std::nullopt)[0]));

    #     # Append vector to total vector of element data for this `obs_id`
    #     element_data.insert(element_data.end(),
    #                         std::make_move_iterator(data_by_element.begin()),
    #                         std::make_move_iterator(data_by_element.end()))
    #     data_by_element.clear()
    #     original_file.close_current_object()

    #     new_file = spectre_h5.H5File(output, "a")
    #     new_volume_file = new_file.get_vol("/" + subfile_name)
    #     new_volume_file.write_volume_data(obs_id, obs_val, element_data,
    #                                     serialized_domain,
    #                                     serialized_functions_of_time)


if __name__ == "__main__":
    combine_h5_vol_command(help_option_names=["-h", "--help"])
