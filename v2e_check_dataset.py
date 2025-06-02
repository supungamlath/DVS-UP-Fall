import argparse
from pathlib import Path
from collections import defaultdict
import re

from utils import load_config


def find_missing_h5_files(dataset_root):
    """
    Find folders that are missing .h5 files in the v2e dataset

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
                avi_files = list(item.glob("*.avi"))
                h5_files = list(item.glob("*.h5"))
                txt_files = list(item.glob("*.txt"))

                # Separate different types of txt files
                frame_times_files = list(item.glob("*frame_times.txt"))
                script_args_files = list(item.glob("*script-args.txt"))
                other_txt_files = [f for f in txt_files if f not in frame_times_files + script_args_files]

                folder_analysis[folder_name] = {
                    "path": item,
                    "avi_count": len(avi_files),
                    "h5_count": len(h5_files),
                    "frame_times_count": len(frame_times_files),
                    "script_args_count": len(script_args_files),
                    "other_txt_count": len(other_txt_files),
                    "avi_files": [f.name for f in avi_files],
                    "h5_files": [f.name for f in h5_files],
                    "frame_times_files": [f.name for f in frame_times_files],
                    "script_args_files": [f.name for f in script_args_files],
                    "other_txt_files": [f.name for f in other_txt_files],
                    "has_h5": len(h5_files) > 0,
                    "has_avi": len(avi_files) > 0,
                    "has_frame_times": len(frame_times_files) > 0,
                    "has_script_args": len(script_args_files) > 0,
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
    Analyze the folder data to identify missing files

    Args:
        folder_analysis (dict): Results from find_missing_h5_files

    Returns:
        dict: Analysis summary
    """
    if not folder_analysis:
        return None

    missing_h5 = []
    missing_avi = []
    missing_frame_times = []
    missing_script_args = []
    incomplete_folders = []

    folders_with_h5 = []
    folders_without_h5 = []

    # Group by subject-activity-trial to identify missing cameras
    subject_activity_trial_groups = defaultdict(list)

    for folder_name, data in folder_analysis.items():
        key = f"Subject{data['subject']}Activity{data['activity']}Trial{data['trial']}"
        subject_activity_trial_groups[key].append(
            {
                "folder": folder_name,
                "camera": data["camera"],
                "has_h5": data["has_h5"],
                "has_avi": data["has_avi"],
                "has_frame_times": data["has_frame_times"],
                "has_script_args": data["has_script_args"],
                "data": data,
            }
        )

        # Track missing files
        if not data["has_h5"]:
            missing_h5.append(data)
            folders_without_h5.append(folder_name)
        else:
            folders_with_h5.append(folder_name)

        if not data["has_avi"]:
            missing_avi.append(data)

        if not data["has_frame_times"]:
            missing_frame_times.append(data)

        if not data["has_script_args"]:
            missing_script_args.append(data)

        # Check if folder has all expected files
        if not (data["has_h5"] and data["has_avi"] and data["has_frame_times"] and data["has_script_args"]):
            incomplete_folders.append(data)

    return {
        "missing_h5": missing_h5,
        "missing_avi": missing_avi,
        "missing_frame_times": missing_frame_times,
        "missing_script_args": missing_script_args,
        "incomplete_folders": incomplete_folders,
        "folders_with_h5": folders_with_h5,
        "folders_without_h5": folders_without_h5,
        "subject_activity_trial_groups": dict(subject_activity_trial_groups),
        "total_folders": len(folder_analysis),
        "folders_missing_h5": len(folders_without_h5),
        "folders_with_h5_count": len(folders_with_h5),
        "folders_missing_avi": len(missing_avi),
        "folders_missing_frame_times": len(missing_frame_times),
        "folders_missing_script_args": len(missing_script_args),
        "incomplete_folders_count": len(incomplete_folders),
    }


def print_detailed_report(analysis):
    """
    Print a detailed report of missing files

    Args:
        analysis (dict): Analysis results
    """
    if not analysis:
        return

    print("=" * 80)
    print("MISSING FILES ANALYSIS REPORT (v2e Dataset)")
    print("=" * 80)

    print(f"\nSUMMARY:")
    print(f"- Total folders analyzed: {analysis['total_folders']}")
    print(f"- Complete folders (all 4 files): {analysis['total_folders'] - analysis['incomplete_folders_count']}")
    print(f"- Incomplete folders: {analysis['incomplete_folders_count']}")
    print(f"\nFILE TYPE BREAKDOWN:")
    print(f"- Folders with .h5 files: {analysis['folders_with_h5_count']}")
    print(f"- Folders missing .h5 files: {analysis['folders_missing_h5']}")
    print(f"- Folders missing .avi files: {analysis['folders_missing_avi']}")
    print(f"- Folders missing frame_times.txt: {analysis['folders_missing_frame_times']}")
    print(f"- Folders missing script-args.txt: {analysis['folders_missing_script_args']}")

    if analysis["folders_missing_h5"] > 0:
        print(f"\n🚨 FOLDERS MISSING .H5 FILES:")
        print("-" * 50)

        for missing in analysis["missing_h5"]:
            folder_path = missing["path"]
            print(f"\n📁 {missing['path'].name}")
            print(f"   Path: {folder_path}")
            print(f"   AVI files: {missing['avi_count']} - {missing['avi_files']}")
            print(f"   H5 files: {missing['h5_count']} - {missing['h5_files']}")
            print(f"   Frame times: {missing['frame_times_count']} - {missing['frame_times_files']}")
            print(f"   Script args: {missing['script_args_count']} - {missing['script_args_files']}")

    if analysis["incomplete_folders_count"] > 0:
        print(f"\n📋 ALL INCOMPLETE FOLDERS:")
        print("-" * 50)

        for incomplete in analysis["incomplete_folders"]:
            print(f"\n📁 {incomplete['path'].name}")
            print(f"   Path: {incomplete['path']}")

            # Show what's missing
            missing_items = []
            if not incomplete["has_h5"]:
                missing_items.append("❌ .h5 file")
            else:
                missing_items.append("✅ .h5 file")

            if not incomplete["has_avi"]:
                missing_items.append("❌ .avi file")
            else:
                missing_items.append("✅ .avi file")

            if not incomplete["has_frame_times"]:
                missing_items.append("❌ frame_times.txt")
            else:
                missing_items.append("✅ frame_times.txt")

            if not incomplete["has_script_args"]:
                missing_items.append("❌ script-args.txt")
            else:
                missing_items.append("✅ script-args.txt")

            print(f"   Status: {' | '.join(missing_items)}")

    # Check for incomplete camera sets per subject-activity-trial
    print(f"\nCOMPLETENESS CHECK BY SUBJECT-ACTIVITY-TRIAL:")
    print("-" * 60)

    incomplete_groups = []

    for group_key, cameras in analysis["subject_activity_trial_groups"].items():
        cameras_with_h5 = [c for c in cameras if c["has_h5"]]
        cameras_without_h5 = [c for c in cameras if not c["has_h5"]]

        if cameras_without_h5:
            incomplete_groups.append(
                {
                    "group": group_key,
                    "total_cameras": len(cameras),
                    "cameras_with_h5": len(cameras_with_h5),
                    "cameras_without_h5": len(cameras_without_h5),
                    "missing_cameras": cameras_without_h5,
                }
            )

    if incomplete_groups:
        for group in incomplete_groups:
            print(f"\n🔍 {group['group']}")
            print(f"   Total cameras: {group['total_cameras']}")
            print(f"   Cameras with .h5: {group['cameras_with_h5']}")
            print(f"   Cameras missing .h5: {group['cameras_without_h5']}")
            for missing_cam in group["missing_cameras"]:
                print(f"     - {missing_cam['folder']} (Camera{missing_cam['camera']})")
    else:
        print("✅ All subject-activity-trial groups have complete camera sets with .h5 files!")


def export_missing_lists(analysis, output_prefix="missing_files_v2e"):
    """
    Export lists of missing files to text files

    Args:
        analysis (dict): Analysis results
        output_prefix (str): Prefix for output files
    """
    if not analysis:
        print("No analysis data to export.")
        return

    # Export missing .h5 files
    if analysis["folders_missing_h5"] > 0:
        h5_file = f"{output_prefix}_h5.txt"
        with open(h5_file, "w") as f:
            f.write("FOLDERS MISSING .H5 FILES\n")
            f.write("=" * 40 + "\n\n")

            for missing in analysis["missing_h5"]:
                f.write(f"{missing['path'].name}\n")
                f.write(f"Path: {missing['path']}\n")
                f.write("-" * 40 + "\n")

        print(f"Missing .h5 files list exported to: {h5_file}")

    # Export all incomplete folders
    if analysis["incomplete_folders_count"] > 0:
        incomplete_file = f"{output_prefix}_incomplete.txt"
        with open(incomplete_file, "w") as f:
            f.write("INCOMPLETE FOLDERS (MISSING ANY FILE TYPE)\n")
            f.write("=" * 50 + "\n\n")

            for incomplete in analysis["incomplete_folders"]:
                f.write(f"{incomplete['path'].name}\n")
                f.write(f"Path: {incomplete['path']}\n")
                f.write(f"H5: {'✅' if incomplete['has_h5'] else '❌'} | ")
                f.write(f"AVI: {'✅' if incomplete['has_avi'] else '❌'} | ")
                f.write(f"Frame Times: {'✅' if incomplete['has_frame_times'] else '❌'} | ")
                f.write(f"Script Args: {'✅' if incomplete['has_script_args'] else '❌'}\n")
                f.write("-" * 50 + "\n")

        print(f"Incomplete folders list exported to: {incomplete_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze v2e dataset for missing files")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    dataset_root = config["paths"]["v2e_dataset_dir"]

    print("Analyzing v2e dataset for missing files...")
    print(f"Dataset path: {dataset_root}")

    # Find missing files
    folder_analysis = find_missing_h5_files(dataset_root)

    if folder_analysis:
        # Analyze the results
        analysis = analyze_missing_files(folder_analysis)

        # Print detailed report
        print_detailed_report(analysis)

        # Ask if user wants to export the lists
        if analysis and (analysis["folders_missing_h5"] > 0 or analysis["incomplete_folders_count"] > 0):
            export_choice = input("\nExport missing files lists to text files? (y/N): ").strip().lower()
            if export_choice in ["y", "yes"]:
                export_missing_lists(analysis)
    else:
        print("Could not analyze dataset.")
