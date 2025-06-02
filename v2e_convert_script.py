import argparse
import os
from pathlib import Path
import subprocess
import zipfile
import shutil
from datetime import datetime

from utils import load_config


def get_zip_files(root_folder):
    zip_files = []
    # Walk through the directory structure starting from root_folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("Subject") and filename.endswith(".zip"):
                # If the file is a .zip file, append its path to zip_files list
                zip_file_path = os.path.join(dirpath, filename)
                zip_files.append(zip_file_path)
    return zip_files


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="V2E Conversion Script")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    up_fall_dataset_dir = Path(config["paths"]["up_fall_dataset_dir"])
    dataset_dir = Path(config["paths"]["dataset_dir"])
    temp_dir = Path(config["paths"]["temp_dir"])

    output_folder = dataset_dir

    # Get image folders list
    zipped_image_folders = get_zip_files(up_fall_dataset_dir)
    print(len(zipped_image_folders))

    # Create extraction directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)

    for zip_file in zipped_image_folders:
        # Get folder name from zip file (using Windows path separator)
        folder_name = os.path.basename(zip_file).replace(".zip", "")

        out_folder = os.path.join(output_folder, folder_name)
        if os.path.exists(out_folder):
            continue

        unzip_path = os.path.join(temp_dir, folder_name)
        os.makedirs(unzip_path, exist_ok=True)

        # Extract zip file using Python's zipfile module
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        # List all files in the folder
        image_files = os.listdir(unzip_path)

        # Extract the timestamps from the filenames
        timestamps = [
            datetime.strptime(file_name.replace(".png", ""), "%Y-%m-%dT%H_%M_%S.%f") for file_name in image_files
        ]
        timestamps.sort()

        # Calculate the duration in seconds
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = (end_time - start_time).total_seconds()
        frame_rate = len(timestamps) / duration if duration > 0 else 0

        print(f"Calculated frame rate: {frame_rate:.2f}")

        os.makedirs(out_folder, exist_ok=True)

        out_filename = f"{folder_name}.h5"
        out_file_path = os.path.join(out_folder, out_filename)

        # Check if output h5 file already exists, skip if it does
        if os.path.exists(out_file_path):
            print(f"Output file {out_file_path} already exists. Skipping...")
            continue

        # Prepare v2e command
        v2e_command = f'v2e -i "{unzip_path}" -o "{out_folder}" --overwrite --vid_orig None --vid_slomo None --unique_output_folder false --ddd_output --dvs_h5 {out_filename} --dvs_aedat2 None --dvs_text None --no_preview --dvs_exposure duration 0.033 --input_frame_rate {frame_rate} --disable_slomo --auto_timestamp_resolution false --timestamp_resolution 0.001 --output_height 120 --output_width 160 --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.02 --cutoff_hz 0 --leak_rate_hz 0 --shot_noise_rate_hz 0'

        print(v2e_command)

        # Run v2e command directly
        result = subprocess.run(v2e_command, shell=True, capture_output=True, text=True)

        # Remove the temporary extraction directory
        shutil.rmtree(unzip_path)
