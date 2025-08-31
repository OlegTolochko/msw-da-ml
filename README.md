# Reimplementation: Training a Convolutional Neural Network to Conserve Mass in Data Assimilation

This repository contains a Python reimplementation of the paper:

**"Training a convolutional neural network to conserve mass in data assimilation"**
by Ruckstuhl, Y. and Janji\'c, T. and Rasp, S.

*Paper: [doi.org/10.5194/npg-28-111-2021](https://doi.org/10.5194/npg-28-111-2021)*

The original authors code can be found at: [zenodo.org/records/4354602](https://zenodo.org/records/4354602)

## Quick Start

### Setup
```bash
git clone https://github.com/OlegTolochko/msw-da-ml.git
cd msw-da-ml/
pip install -e .  # Install in development mode
```

### Running the Pipeline

1. **Generate training data:**
   ```bash
   python -m msw_da_ml.main generate-training-data
   ```

2. **Train a CNN model:**
   ```bash
   python -m msw_da_ml.main train-cnn-model <training_data_name>
   ```

3. **Run inference:**
   ```bash
   python -m msw_da_ml.main run-inference
   ```

4. **Generate experiment data for uncertainty quantification:**
   ```bash
   python -m msw_da_ml.main generate-conformal-prediction-data
   ```

5. **Compare different uncertainty quantification methods:**
   ```bash
   python -m msw_da_ml.main compare-cp-vs-cqr <cp_data> <cqr_data>
   ```

### Useful Commands

```bash
# Show all available commands
python -m msw_da_ml.main --help

# List trained models
python -m msw_da_ml.main list-available-models

# List training/experiment data
python -m msw_da_ml.main list-available-data

# Get detailed help for any command
python -m msw_da_ml.main <command> --help
```

## Configuration

Model parameters and experiment settings can be configured in `config.yaml`.