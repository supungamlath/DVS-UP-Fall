import argparse
import os
import re
from pathlib import Path

from utils import load_config


def rename_npz_files(dataset_root):
    """
    Rename all .npz files in the dataset to format: SubjectXActivityYTrialZCameraW.npz

    Args:
        dataset_root (str): Path to the root directory of the dataset
    """
    dataset_path = Path(dataset_root)

    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_root}' does not exist.")
        return

    # Counter for tracking operations
    renamed_count = 0
    error_count = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".npz"):
                old_filepath = Path(root) / file

                # Extract subject, activity, trial, camera info from filename
                # Pattern: Subject17Activity10Trial3Camera1-ceil_10-fps_18-events.npz
                pattern = r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)"
                match = re.search(pattern, file)

                if match:
                    subject_num = match.group(1)
                    activity_num = match.group(2)
                    trial_num = match.group(3)
                    camera_num = match.group(4)

                    # Create new filename
                    new_filename = f"Subject{subject_num}Activity{activity_num}Trial{trial_num}Camera{camera_num}.npz"
                    new_filepath = old_filepath.parent / new_filename

                    # Check if new filename is different from old
                    if old_filepath.name != new_filename:
                        try:
                            # Check if target file already exists
                            if new_filepath.exists():
                                print(f"Warning: Target file already exists, skipping: {new_filepath}")
                                continue

                            # Rename the file
                            old_filepath.rename(new_filepath)
                            print(f"Renamed: {old_filepath.name} -> {new_filename}")
                            renamed_count += 1

                        except Exception as e:
                            print(f"Error renaming {old_filepath}: {e}")
                            error_count += 1
                    else:
                        print(f"Already correct format: {file}")
                else:
                    print(f"Warning: Could not extract info from filename: {file}")
                    error_count += 1

    # Summary
    print(f"\nOperation completed:")
    print(f"- Files renamed: {renamed_count}")
    print(f"- Errors/Warnings: {error_count}")


def preview_changes(dataset_root):
    """
    Preview what changes would be made without actually renaming files

    Args:
        dataset_root (str): Path to the root directory of the dataset
    """
    dataset_path = Path(dataset_root)

    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_root}' does not exist.")
        return

    print("Preview of changes (no files will be renamed):")
    print("-" * 60)

    file_count = 0
    changes_count = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".npz"):
                file_count += 1
                # Extract subject, activity, trial, camera info from filename
                pattern = r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)"
                match = re.search(pattern, file)

                if match:
                    subject_num = match.group(1)
                    activity_num = match.group(2)
                    trial_num = match.group(3)
                    camera_num = match.group(4)

                    new_filename = f"Subject{subject_num}Activity{activity_num}Trial{trial_num}Camera{camera_num}.npz"

                    if file != new_filename:
                        rel_path = Path(root).relative_to(dataset_path)
                        print(f"{rel_path}/{file}")
                        print(f"  -> {new_filename}")
                        changes_count += 1

    print(f"\nTotal files found: {file_count}")
    print(f"\nTotal files to be renamed: {changes_count}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze v2ce dataset for missing files")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    dataset_root = config["paths"]["v2ce_dataset_dir"]

    # First, preview the changes
    print("=== PREVIEW MODE ===")
    preview_changes(dataset_root)

    # Ask for confirmation
    response = input("\nProceed with renaming? (y/N): ").strip().lower()

    if response == "y" or response == "yes":
        print("\n=== RENAMING FILES ===")
        rename_npz_files(dataset_root)
    else:
        print("Operation cancelled.")
