import argparse
from pathlib import Path
from collections import defaultdict
import re

from utils import load_config


def find_missing_npz_files(dataset_root):
    """
    Find folders that are missing .npz files in the dataset

    Args:
        dataset_root (str): Path to the root directory of the dataset

    Returns:
        dict: Analysis results with missing files information
    """
    dataset_path = Path(dataset_root)

    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_root}' does not exist.")
        return None

    # Dictionary to store folder analysis
    folder_analysis = {}

    # Pattern to match folder names: SubjectXActivityYTrialZCameraW
    folder_pattern = r"Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)"

    # Walk through all subdirectories
    for item in dataset_path.iterdir():
        if item.is_dir():
            folder_name = item.name

            # Check if folder matches expected pattern
            match = re.search(folder_pattern, folder_name)
            if match:
                # Count files in this folder
                mp4_files = list(item.glob("*.mp4"))
                npz_files = list(item.glob("*.npz"))

                folder_analysis[folder_name] = {
                    "path": item,
                    "mp4_count": len(mp4_files),
                    "npz_count": len(npz_files),
                    "mp4_files": [f.name for f in mp4_files],
                    "npz_files": [f.name for f in npz_files],
                    "has_npz": len(npz_files) > 0,
                    "subject": match.group(1),
                    "activity": match.group(2),
                    "trial": match.group(3),
                    "camera": match.group(4),
                }
            else:
                print(f"Warning: Folder '{folder_name}' doesn't match expected pattern")

    return folder_analysis


def analyze_missing_files(folder_analysis):
    """
    Analyze the folder data to identify missing .npz files

    Args:
        folder_analysis (dict): Results from find_missing_npz_files

    Returns:
        dict: Analysis summary
    """
    if not folder_analysis:
        return None

    missing_npz = []
    folders_with_npz = []
    folders_without_npz = []

    # Group by subject-activity-trial to identify missing cameras
    subject_activity_trial_groups = defaultdict(list)

    for folder_name, data in folder_analysis.items():
        key = f"Subject{data['subject']}Activity{data['activity']}Trial{data['trial']}"
        subject_activity_trial_groups[key].append(
            {"folder": folder_name, "camera": data["camera"], "has_npz": data["has_npz"], "data": data}
        )

        if data["has_npz"]:
            folders_with_npz.append(folder_name)
        else:
            folders_without_npz.append(folder_name)
            missing_npz.append(data)

    return {
        "missing_npz": missing_npz,
        "folders_with_npz": folders_with_npz,
        "folders_without_npz": folders_without_npz,
        "subject_activity_trial_groups": dict(subject_activity_trial_groups),
        "total_folders": len(folder_analysis),
        "folders_missing_npz": len(folders_without_npz),
        "folders_with_npz_count": len(folders_with_npz),
    }


def print_detailed_report(analysis):
    """
    Print a detailed report of missing .npz files

    Args:
        analysis (dict): Analysis results
    """
    if not analysis:
        return

    print("=" * 80)
    print("MISSING .NPZ FILES ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nSUMMARY:")
    print(f"- Total folders analyzed: {analysis['total_folders']}")
    print(f"- Folders with .npz files: {analysis['folders_with_npz_count']}")
    print(f"- Folders missing .npz files: {analysis['folders_missing_npz']}")

    if analysis["folders_missing_npz"] > 0:
        print(f"\nFOLDERS MISSING .NPZ FILES:")
        print("-" * 50)

        for missing in analysis["missing_npz"]:
            folder_path = missing["path"]
            print(f"\n📁 {missing['path'].name}")
            print(f"   Path: {folder_path}")
            print(f"   MP4 files: {missing['mp4_count']} - {missing['mp4_files']}")
            print(f"   NPZ files: {missing['npz_count']} - {missing['npz_files']}")

    # Check for incomplete camera sets per subject-activity-trial
    print(f"\nCOMPLETENESS CHECK BY SUBJECT-ACTIVITY-TRIAL:")
    print("-" * 60)

    incomplete_groups = []

    for group_key, cameras in analysis["subject_activity_trial_groups"].items():
        cameras_with_npz = [c for c in cameras if c["has_npz"]]
        cameras_without_npz = [c for c in cameras if not c["has_npz"]]

        if cameras_without_npz:
            incomplete_groups.append(
                {
                    "group": group_key,
                    "total_cameras": len(cameras),
                    "cameras_with_npz": len(cameras_with_npz),
                    "cameras_without_npz": len(cameras_without_npz),
                    "missing_cameras": cameras_without_npz,
                }
            )

    if incomplete_groups:
        for group in incomplete_groups:
            print(f"\n🔍 {group['group']}")
            print(f"   Total cameras: {group['total_cameras']}")
            print(f"   Cameras with .npz: {group['cameras_with_npz']}")
            print(f"   Cameras missing .npz: {group['cameras_without_npz']}")
            for missing_cam in group["missing_cameras"]:
                print(f"     - {missing_cam['folder']} (Camera{missing_cam['camera']})")
    else:
        print("✅ All subject-activity-trial groups have complete camera sets with .npz files!")


def export_missing_list(analysis, output_file="missing_npz_files.txt"):
    """
    Export list of missing .npz files to a text file

    Args:
        analysis (dict): Analysis results
        output_file (str): Output file name
    """
    if not analysis or analysis["folders_missing_npz"] == 0:
        print("No missing files to export.")
        return

    with open(output_file, "w") as f:
        f.write("FOLDERS MISSING .NPZ FILES\n")
        f.write("=" * 40 + "\n\n")

        for missing in analysis["missing_npz"]:
            f.write(f"{missing['path'].name}\n")
            f.write(f"Path: {missing['path']}\n")
            f.write(f"MP4 files: {missing['mp4_files']}\n")
            f.write(f"NPZ files: {missing['npz_files']}\n")
            f.write("-" * 40 + "\n")

    print(f"Missing files list exported to: {output_file}")


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

    print("Analyzing dataset for missing .npz files...")
    print(f"Dataset path: {dataset_root}")

    # Find missing .npz files
    folder_analysis = find_missing_npz_files(dataset_root)

    if folder_analysis:
        # Analyze the results
        analysis = analyze_missing_files(folder_analysis)

        # Print detailed report
        print_detailed_report(analysis)

        # Ask if user wants to export the list
        if analysis and analysis["folders_missing_npz"] > 0:
            export_choice = input("\nExport missing files list to text file? (y/N): ").strip().lower()
            if export_choice in ["y", "yes"]:
                export_missing_list(analysis)
    else:
        print("Could not analyze dataset.")
