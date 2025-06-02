import os
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py


class SpikingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        time_duration=60.0,
        label_resolution=1.0,
        read_csv=True,
        camera1_only=False,
        multiclass=False,
    ):
        self.root_dir = root_dir
        self.time_duration = time_duration
        self.max_timestamp = time_duration * 1e6
        self.frame_height = 120
        self.frame_width = 160
        self.scaling_factor = 1.0
        self.label_resolution = label_resolution
        self.camera1_only = camera1_only

        if multiclass:
            self.labels_csv = f"labels_multiclass_w{label_resolution:.1f}.csv"
        else:
            self.labels_csv = f"labels_binary_w{label_resolution:.1f}.csv"

        if read_csv:
            labels = self.get_fall_flags()
            self.folder_names = list(labels.keys())
            self.labels = list(labels.values())

    def get_fall_flags(self):
        """
        Reads the labels CSV file and extracts the fall flag values for each video.

        Returns:
            dict: A dictionary where keys are video names and values are lists of fall flags for each window.
        """
        # Read the CSV file
        labels_file = os.path.join(self.root_dir, self.labels_csv)
        df = pd.read_csv(labels_file)

        # Initialize the result dictionary
        fall_flags_dict = {}

        window_range = 60
        if self.label_resolution == 0.1:
            window_range = 599
        elif self.label_resolution == 0.5:
            window_range = 120
        elif self.label_resolution == 1.0:
            window_range = 60
        else:
            raise ValueError("Invalid label resolution.")

        # Iterate over each row
        for _, row in df.iterrows():
            # Extract video name and fall flags
            video_name = row["name"]
            if self.camera1_only and "Camera1" not in video_name:
                continue
            fall_flags = [row[f"window_{i+1}"] for i in range(window_range)]

            # Add to the dictionary
            fall_flags_dict[video_name] = fall_flags

        return fall_flags_dict

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        events_folder = self.folder_names[idx]

        events_file_path = Path(os.path.join(self.root_dir, events_folder, events_folder + ".h5"))
        if events_file_path.exists():
            with h5py.File(events_file_path, "r") as spike_data:
                spike_tuples = np.array(spike_data["events"])
        else:
            events_file_path = Path(os.path.join(self.root_dir, events_folder, events_folder + ".npz"))
            spike_data = np.load(events_file_path)
            spike_tuples = np.array([list(spike) for spike in spike_data["event_stream"]])

        # Remove events that occur after max_timestamp
        spike_tuples = spike_tuples[spike_tuples[:, 0] < self.max_timestamp]
        spike_array = np.zeros(
            spike_tuples.shape[0],
            dtype=[
                ("t", np.float32),
                ("x", np.int32),
                ("y", np.int32),
                ("p", np.int32),
            ],
        )
        spike_array["t"] = spike_tuples[:, 0] / 1e6 * self.scaling_factor
        spike_array["x"] = spike_tuples[:, 1]
        spike_array["y"] = spike_tuples[:, 2]
        spike_array["p"] = spike_tuples[:, 3]
        return spike_array, self.labels[idx]

    def edit_label(self, idx, label):
        self.labels[idx] = label

    def get_folder_name(self, idx):
        return self.folder_names[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_video_path(self, idx):
        video_path = os.path.join(self.root_dir, self.folder_names[idx], "dvs-video.avi")
        return os.path.normpath(video_path)

    def random_split(self, test_size=0.25, shuffle=True):
        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )

        # Ensure reproducibility
        np.random.seed(42)

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Shuffle combined data if required
        if shuffle:
            np.random.shuffle(combined)

        # Calculate the split index
        total_elements = len(combined)
        test_size_elements = int(total_elements * test_size)

        # Split the data
        train_data = combined[:-test_size_elements]
        test_data = combined[-test_size_elements:]

        # Separate folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data) if train_data else ([], [])
        test_dataset.folder_names, test_dataset.labels = zip(*test_data) if test_data else ([], [])

        return train_dataset, test_dataset

    def split_by_subjects(self):
        """
        Splits the dataset into training, development, and test subsets based on predefined subjects.
        """

        train_subjects = {
            "Subject1",
            "Subject3",
            "Subject4",
            "Subject7",
            "Subject10",
            "Subject11",
            "Subject12",
            "Subject13",
            "Subject14",
        }
        test_subjects = {"Subject15", "Subject16", "Subject17"}
        dev_subjects = {"Subject2", "Subject5", "Subject6", "Subject8", "Subject9"}

        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )
        dev_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Separate data based on subjects
        train_data = []
        dev_data = []
        test_data = []

        for folder_name, label in combined:
            subject = folder_name.split("Activity")[0]
            if subject in train_subjects:
                train_data.append((folder_name, label))
            elif subject in test_subjects:
                test_data.append((folder_name, label))
            elif subject in dev_subjects:
                dev_data.append((folder_name, label))

        # Raise error if any dataset is empty
        if not train_data:
            raise ValueError("Training dataset is empty!")
        if not dev_data:
            raise ValueError("Development dataset is empty!")
        if not test_data:
            raise ValueError("Test dataset is empty!")

        # Separate the folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data)
        dev_dataset.folder_names, dev_dataset.labels = zip(*dev_data)
        test_dataset.folder_names, test_dataset.labels = zip(*test_data)

        return train_dataset, dev_dataset, test_dataset

    def split_by_trials(self):
        """
        Splits the dataset into training, development, and test sets based on trial information.

        - Data from trial 3 is used exclusively for the test set.
        - Data from trial 2 for subjects 1, 3, and 4 is used for the development set.
        - All other data is used for the training set.
        """

        # Initialize datasets
        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )
        dev_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            label_resolution=self.label_resolution,
            read_csv=False,
        )

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Separate data based on trials
        train_data = []
        dev_data = []
        test_data = []

        # Define subjects for the development set
        dev_subjects = {"Subject1", "Subject3", "Subject4"}

        for folder_name, label in combined:
            subject = folder_name.split("Activity")[0]

            if "Trial3" in folder_name:
                test_data.append((folder_name, label))
            elif "Trial2" in folder_name and subject in dev_subjects:
                dev_data.append((folder_name, label))
            else:
                train_data.append((folder_name, label))

        # Raise error if any dataset is empty
        if not train_data:
            raise ValueError("Training dataset is empty!")
        if not dev_data:
            raise ValueError("Development dataset is empty!")
        if not test_data:
            raise ValueError("Test dataset is empty!")

        # Separate the folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data)
        dev_dataset.folder_names, dev_dataset.labels = zip(*dev_data)
        test_dataset.folder_names, test_dataset.labels = zip(*test_data)

        return train_dataset, dev_dataset, test_dataset

    def save_labels(self):
        df = pd.DataFrame(
            {
                "folder_name": self.folder_names,
                "label": self.labels,
            }
        )
        df.to_csv(
            os.path.join(self.root_dir, self.labels_csv),
            index=False,
        )
