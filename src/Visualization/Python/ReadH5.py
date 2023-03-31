# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py


def available_subfiles(h5file, extension):
    """List all subfiles with the given 'extension' in the 'h5file'.

    Parameters
    ----------
    h5file: Open h5py file
    extension: str

    Returns
    -------
    List of paths in the 'h5file' that end with the 'extension'
    """
    subfiles = []

    def visitor(name):
        if name.endswith(extension):
            subfiles.append(name)

    h5file.visit(visitor)
    return subfiles


def to_dataframe(open_subfile):
    """Convert a '.dat' subfile to a Pandas DataFrame

    This function isn't particularly complex, but it allows to convert a
    subfile to a DataFrame in a single statement like this:

        to_dataframe(open_h5_file["Norms.dat"])

    Without this function, you would have to store the subfile in an extra
    variable to access its "Legend" attribute.

    Arguments:
      open_subfile: An open h5py subfile representing a SpECTRE dat file,
        typically from a reductions file.

    Returns: Pandas DataFrame with column names read from the "Legend"
      attribute of the dat file.
    """
    import pandas as pd
    return pd.DataFrame(open_subfile, columns=open_subfile.attrs["Legend"])
