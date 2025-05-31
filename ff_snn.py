import argparse
from pathlib import Path
import torch
from norse.torch import LICell, LIFCell, LIFParameters
from tqdm import tqdm, trange

from SpikingDataset import SpikingDataset
from SpikingDataLoader import SpikingDataLoader
from Metrics import Metrics
from utils import load_config


class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, tau_mem, tau_syn):
        super(SNN, self).__init__()
        self.fc_hidden = torch.nn.Linear(input_features, hidden_features, bias=False)
        self.cell = LIFCell(
            p=LIFParameters(
                tau_mem_inv=torch.tensor(1 / (tau_mem * 0.001)),
                tau_syn_inv=torch.tensor(1 / (tau_syn * 0.001)),
                alpha=100,
            ),
        )
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(
            p=LIFParameters(
                tau_mem_inv=torch.tensor(1 / (tau_mem * 0.001)),
                tau_syn_inv=torch.tensor(1 / (tau_syn * 0.001)),
                alpha=100,
            ),
        )
        self.input_features = input_features

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        for ts in range(seq_length):
            z = x[ts].view(-1, self.input_features)
            z = self.fc_hidden(z)
            z, s1 = self.cell(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            voltages += [vo]

        return torch.stack(voltages)


class Model(torch.nn.Module):
    def __init__(self, snn, nb_steps, nb_labels):
        super(Model, self).__init__()
        self.snn = snn
        self.nb_steps = nb_steps
        self.nb_labels = nb_labels

    def decode(self, x):
        # x: (T, B, 12)
        nb_steps, batch_size, output_size = x.shape
        steps_per_label = nb_steps // self.nb_labels
        x = x.view(self.nb_labels, steps_per_label, batch_size, output_size)
        x, _ = torch.max(x, 1)
        return x  # (T, B, 12)

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decode(x)
        return log_p_y


def train(model, device, train_loader, optimizer, loss_fn, metrics):
    model.train()

    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to_dense().transpose(0, 1))
        loss = loss_fn(output.permute(1, 2, 0), target.long())
        pred = output.argmax(dim=2).permute(1, 0)
        metrics.update(pred.cpu().detach().numpy(), target.cpu().detach().numpy(), loss.item())
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, loss_fn, metrics):
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data.to_dense().transpose(0, 1))
            loss = loss_fn(output.permute(1, 2, 0), target.long())
            pred = output.argmax(dim=2).permute(1, 0)
            metrics.update(pred.cpu().detach().numpy(), target.cpu().detach().numpy(), loss.item())


def run_training(model, device, train_loader, dev_loader, test_loader, optimizer, loss_fn, epochs):
    train_metrics = Metrics()
    dev_metrics = Metrics()
    test_metrics = Metrics()

    torch.autograd.set_detect_anomaly(True)

    for epoch in trange(epochs):
        train(model, device, train_loader, optimizer, loss_fn, train_metrics)
        train_metrics_dict = train_metrics.compute()
        print(f"Epoch {epoch+1}/{epochs} \nTrain Metrics: {train_metrics_dict}")
        train_metrics.reset()

        test(model, device, dev_loader, loss_fn, dev_metrics)
        dev_metrics_dict = dev_metrics.compute()
        print(f"Epoch {epoch+1}/{epochs} \nDev Metrics: {dev_metrics_dict}")
        dev_metrics.reset()

    test(model, device, test_loader, loss_fn, test_metrics)
    test_metrics_dict = test_metrics.compute()
    test_metrics.reset()

    print(f"Test Metrics: {test_metrics_dict}")
    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SNN Training Script")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    # Extract paths and project configuration
    dataset_dir = Path(config["paths"]["dataset_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    model_name = config["clearml"]["model_name"]

    # Read model parameters
    model_params = {
        "tau_mem": 8.0,
        "tau_syn": 16.0,
        "input_features": 2 * 40 * 30,
        "hidden_features": 1000,
        "multiclass": False,
    }
    last_layer_size = 12 if model_params["multiclass"] else 2

    # Read training parameters
    training_params = {
        "learning_rate": 0.002,
        "nb_epochs": 20,
        "nb_steps": 120,
        "batch_size": 64,
        "early_stopping": False,
    }

    # Read dataset parameters
    dataset_params = {
        "target_height": 30,
        "target_width": 40,
        "time_duration": 60.0,
        "label_resolution": 0.5,
        "bias_ratio": 5.0,
        "camera1_only": False,
        "split_by": "subjects",  # "subjects" or "trials"
    }

    # Set device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")

    # Load dataset
    dataset = SpikingDataset(
        root_dir=dataset_dir,
        time_duration=dataset_params["time_duration"],
        label_resolution=dataset_params["label_resolution"],
        camera1_only=dataset_params["camera1_only"],
        multiclass=model_params["multiclass"],
    )

    # Splitting the dataset
    if dataset_params["split_by"] == "subjects":
        train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects()
    elif dataset_params["split_by"] == "trials":
        train_dataset, dev_dataset, test_dataset = dataset.split_by_trials()
    else:
        raise ValueError("Invalid value for split_by parameter")

    # Creating DataLoaders
    train_loader = SpikingDataLoader(
        dataset=train_dataset,
        nb_steps=training_params["nb_steps"],
        batch_size=training_params["batch_size"],
        target_height=dataset_params["target_height"],
        target_width=dataset_params["target_width"],
        shuffle=False,
    )
    dev_loader = SpikingDataLoader(
        dataset=dev_dataset,
        nb_steps=training_params["nb_steps"],
        batch_size=training_params["batch_size"],
        target_height=dataset_params["target_height"],
        target_width=dataset_params["target_width"],
        shuffle=False,
    )
    test_loader = SpikingDataLoader(
        dataset=test_dataset,
        nb_steps=training_params["nb_steps"],
        batch_size=training_params["batch_size"],
        target_height=dataset_params["target_height"],
        target_width=dataset_params["target_width"],
        shuffle=False,
    )
    # Create model
    print("Creating SNN model...")
    model = Model(
        snn=SNN(
            input_features=model_params["input_features"],
            hidden_features=model_params["hidden_features"],
            output_features=last_layer_size,
            tau_mem=model_params["tau_mem"],
            tau_syn=model_params["tau_syn"],
        ),
        nb_steps=training_params["nb_steps"],
        nb_labels=int(dataset_params["time_duration"] / dataset_params["label_resolution"]),
    ).to(DEVICE)
    print(model)

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])

    # Train model
    print("Beginning Model Evaluation...")
    model_after = run_training(
        model, DEVICE, train_loader, dev_loader, test_loader, optimizer, loss_fn, training_params["nb_epochs"]
    )

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_after.state_dict(), model_dir / f"{model_name}.pth")
    print(f"Model saved as '{model_name}.pth'")
