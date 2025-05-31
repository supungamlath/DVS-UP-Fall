# DVS-UP-Fall Dataset Benchmark

This project implements a Spiking Neural Network (SNN) for as a baseline solution for the DVS-UP-Fall dataset, specifically for fall detection using DVS (Dynamic Vision Sensor) data. The codebase includes dataset conversion scripts, visualization tools, and a full SNN training pipeline.

## Project Structure

```
.
├── ff_snn.py               # Main entry point for SNN training and evaluation
├── config sample.yaml      # Sample configuration file for paths and model parameters
├── Metrics.py              # Metrics computation (accuracy, precision, recall, F1 score)
├── SpikingDataset.py       # PyTorch Dataset for DVS event data
├── SpikingDataLoader.py    # Custom DataLoader for sparse event data
├── utils.py                # Utility functions 
├── v2e_script.py           # Script for converting video frames to DVS events (V2E)
├── v2ce_script.py          # Script for converting video frames to events using V2CE
├── visualize_events.py     # Visualization tools for DVS events 
└── requirements.txt        # Python dependencies
```

## Features

- **Custom SNN Architecture**: Implements a simple feedforward SNN with configurable parameters as a baseline solution.
- **Flexible Dataset Handling**: Supports splitting by subjects or trials, and works with both `.h5` and `.npz` event formats.
- **Training & Evaluation**: Includes full training, validation, and test pipelines with detailed metrics.
- **Event Data Conversion**: Scripts for converting video/image sequences to DVS event streams.
- **Visualization**: Tools for visualizing event streams in 2D and 3D.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supungamlath/DVS-UP-Fall
   cd SNN_Fall_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.yaml` to set dataset paths, model directories, and model name. A sample [`config sample.yaml`](config sample.yaml) is provided for reference.

## Usage

1. **Train and Evaluate the SNN:**
   ```bash
   python ff_snn.py --config config.yaml
   ```

   This will:
   - Load the configuration
   - Prepare the dataset and dataloaders
   - Train the SNN model
   - Save the trained model to the specified directory
   - Evaluate on the test set

2. **Convert Video Frames to Events:**
   - Using V2E:
     ```bash
     python v2e_script.py --config config.yaml
     ```
   - Using V2CE:
     ```bash
     git clone https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox
     python v2ce_script.py --config config.yaml
     ```

3. **Visualize Events:**
   ```bash
   python visualize_events.py
   ```
   Edit the file to point to your event data files as needed.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Acknowledgments

- [Norse](https://github.com/norse/norse) for SNN framework with LIF implementation.
- [SNNTorch](https://scikit-learn.org/stable/index.html) for metric calculations.