import argparse
import os
from pathlib import Path
import zipfile
import shutil
from datetime import datetime
from argparse import Namespace

from utils import load_config
from v2ce import main


def get_zip_files(root_folder):
    zip_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("Subject") and filename.endswith(".zip"):
                zip_files.append(os.path.join(dirpath, filename))
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

    zipped_image_folders = get_zip_files(up_fall_dataset_dir)
    print(len(zipped_image_folders))

    extract_base_dir = temp_dir
    os.makedirs(extract_base_dir, exist_ok=True)

    for zip_file in zipped_image_folders:
        folder_name = os.path.basename(zip_file).replace(".zip", "")
        out_folder = os.path.join(output_folder, folder_name)
        if os.path.exists(out_folder):
            continue

        unzip_path = os.path.join(extract_base_dir, folder_name)
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        image_files = os.listdir(unzip_path)
        timestamps = [datetime.strptime(f.replace(".png", ""), "%Y-%m-%dT%H_%M_%S.%f") for f in image_files]
        timestamps.sort()

        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = (end_time - start_time).total_seconds()
        frame_rate = len(timestamps) / duration if duration > 0 else 30

        print(f"Calculated frame rate: {frame_rate:.2f}")

        os.makedirs(out_folder, exist_ok=True)

        # Prepare the args Namespace for calling main()
        args = Namespace(
            fps=int(frame_rate),
            seq_len=120,
            ceil=10,
            upper_bound_percentile=98,
            image_folder=unzip_path,
            input_video_path=None,
            out_folder=out_folder,
            infer_type="center",
            model_path="./weights/v2ce_3d.pt",
            out_name_suffix="",
            max_frame_num=1800,
            width=160,
            height=120,
            write_event_frame_video=True,
            vis_keep_polarity=True,
            log_level="info",
            batch_size=1,
            stage2_batch_size=24,
        )

        # Call the main function directly
        main(args)

        # Clean up extracted files
        shutil.rmtree(unzip_path)
