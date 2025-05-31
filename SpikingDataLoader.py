import numpy as np
import torch
from torch.utils.data import DataLoader


# Custom DataLoader that uses a sparse_data_generator as the collate function
class SpikingDataLoader(DataLoader):
    def __init__(self, dataset, nb_steps, target_width, target_height, *args, **kwargs):
        self.time_duration = dataset.time_duration
        self.nb_steps = nb_steps
        self.target_width = target_width
        self.target_height = target_height
        self.frame_width = dataset.frame_width
        self.frame_height = dataset.frame_height
        self.batch_size = kwargs.get("batch_size")
        print(f"Initializing DataLoader of size {len(dataset)}")

        super().__init__(
            dataset,
            *args,
            collate_fn=self.sparse_data_generator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            **kwargs,
        )

    def sparse_data_generator(
        self,
        batch,
    ):
        time_bins = np.linspace(0, self.time_duration, num=self.nb_steps)

        coo = [[] for _ in range(5)]  # Add polarity as a new coordinate dimension
        labels = []

        for i, datapoint in enumerate(batch):
            sample, label = datapoint
            times = np.digitize(sample["t"], time_bins)
            polarities = sample["p"]  # 0 or 1
            # Scale spatial coordinates to new resolution
            scaled_x = np.floor(sample["x"] * self.target_width / self.frame_width).astype(int)
            scaled_y = np.floor(sample["y"] * self.target_height / self.frame_height).astype(int)

            # Clip to ensure valid indices
            scaled_x = np.clip(scaled_x, 0, self.target_width - 1)
            scaled_y = np.clip(scaled_y, 0, self.target_height - 1)

            batch_indices = [i for _ in range(len(times))]

            coo[0].extend(batch_indices)  # batch index
            coo[1].extend(times)  # time bin index
            coo[2].extend(polarities)  # polarity channel (0 or 1)
            coo[3].extend(scaled_x)
            coo[4].extend(scaled_y)

            labels.append(label)

        i = torch.LongTensor(coo)  # Shape: [5, N_events]
        v = torch.FloatTensor(np.ones(len(coo[0])))  # All values are 1.0

        X_batch = torch.sparse_coo_tensor(
            i, v, torch.Size([self.batch_size, self.nb_steps, 2, self.target_width, self.target_height])
        )

        y_batch = torch.tensor(labels)

        return X_batch, y_batch
