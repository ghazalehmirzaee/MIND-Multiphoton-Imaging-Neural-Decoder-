import scipy.io as sio
import numpy as np
import os


def exclude_cells_from_calcium_signal(file_path):
    """
    Exclude specified cells from calcium signal matrix to match other processed matrices.

    This function loads a MATLAB file, excludes specific neurons from the calcium signal
    matrix based on the excluded_cells list, and saves the result back to the file.

    Parameters:
    file_path (str): Path to the MATLAB .mat file
    """

    print("Loading MATLAB file...")
    # Load the .mat file - this reads all workspaces into a dictionary
    mat_data = sio.loadmat(file_path)

    # Extract the relevant matrices and convert to numpy arrays
    # Note: MATLAB arrays are loaded as numpy arrays automatically
    calciumsignal = mat_data['calciumsignal']
    excluded_cells = mat_data['excluded_cells'].flatten()  # Flatten in case it's a column vector
    deconv_mat = mat_data['DeconvMat_wanted']
    deltaf_mat = mat_data['deltaf_cells_not_excluded']

    print(f"Original calcium signal shape: {calciumsignal.shape}")
    print(f"Number of cells to exclude: {len(excluded_cells)}")
    print(f"Target shape (matching other matrices): {deconv_mat.shape}")

    # Verify our assumptions about the data structure
    assert calciumsignal.shape[0] == deconv_mat.shape[0], "Time dimension mismatch"
    assert calciumsignal.shape[1] - len(excluded_cells) == deconv_mat.shape[1], "Cell count mismatch"

    # MATLAB uses 1-based indexing, but Python uses 0-based indexing
    # We need to convert the excluded cell indices from MATLAB to Python indexing
    excluded_cells_python = excluded_cells - 1  # Convert from 1-based to 0-based indexing

    # Verify that all excluded cell indices are valid
    assert np.all(excluded_cells_python >= 0), "Some excluded cell indices are negative after conversion"
    assert np.all(excluded_cells_python < calciumsignal.shape[1]), "Some excluded cell indices are out of bounds"

    # Create a boolean mask for the cells we want to keep
    # Start with all cells included (True), then set excluded cells to False
    keep_cells_mask = np.ones(calciumsignal.shape[1], dtype=bool)
    keep_cells_mask[excluded_cells_python] = False

    # Apply the mask to exclude the specified cells
    # This keeps all time points (all rows) but only the non-excluded columns (cells)
    calciumsignal_excluded = calciumsignal[:, keep_cells_mask]

    print(f"Resulting calcium signal shape: {calciumsignal_excluded.shape}")

    # Verify that our result matches the expected dimensions
    assert calciumsignal_excluded.shape == deconv_mat.shape, f"Shape mismatch: got {calciumsignal_excluded.shape}, expected {deconv_mat.shape}"

    # Add the new matrix to our data dictionary
    mat_data['calciumsignal_excluded'] = calciumsignal_excluded

    # Save the updated data back to the .mat file
    # We'll create a backup first, then save the updated version
    backup_path = file_path.replace('.mat', '_backup.mat')

    print(f"Creating backup at: {backup_path}")
    sio.savemat(backup_path, mat_data, format='5')

    print(f"Saving updated file with calciumsignal_excluded...")
    sio.savemat(file_path, mat_data, format='5')

    print("Success! The calciumsignal_excluded matrix has been added to your .mat file.")
    print(f"Final matrix shape: {calciumsignal_excluded.shape}")
    print(f"This matches the shape of DeconvMat_wanted and deltaf_cells_not_excluded: {deconv_mat.shape}")

    return calciumsignal_excluded


def verify_cell_alignment(file_path):
    """
    Optional verification function to double-check that the neuron indices align correctly.
    This function can help ensure that the same neurons are represented across all matrices.
    """

    mat_data = sio.loadmat(file_path)

    calc_excluded = mat_data['calciumsignal_excluded']
    deconv_mat = mat_data['DeconvMat_wanted']
    deltaf_mat = mat_data['deltaf_cells_not_excluded']

    print("\nVerification Summary:")
    print(f"calciumsignal_excluded shape: {calc_excluded.shape}")
    print(f"DeconvMat_wanted shape: {deconv_mat.shape}")
    print(f"deltaf_cells_not_excluded shape: {deltaf_mat.shape}")

    # Check if shapes match exactly
    shapes_match = (calc_excluded.shape == deconv_mat.shape == deltaf_mat.shape)
    print(f"All matrix shapes match: {shapes_match}")

    if shapes_match:
        print("✓ Success: All matrices now have matching neuron indices!")
    else:
        print("✗ Warning: Matrix shapes do not match. Please check the exclusion process.")


if __name__ == "__main__":
    # Set the file path
    file_path = "/home/ghazal/Documents/NS_Projects/NS_P2_050325/MIND-Multiphoton-Imaging-Neural-Decoder/data/raw/SFL13_3_80321_010_new.mat"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please verify the file path and try again.")
    else:
        try:
            # Perform the cell exclusion
            result_matrix = exclude_cells_from_calcium_signal(file_path)

            # Run verification
            verify_cell_alignment(file_path)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please check your data file and try again.")

