# Reimplementation: Training a Convolutional Neural Network to Conserve Mass in Data Assimilation

This repository contains a Python reimplementation of the paper:

**"Training a convolutional neural network to conserve mass in data assimilation"**
by Ruckstuhl, Y. and Janji\'c, T. and Rasp, S.

*Paper: [doi.org/10.5194/npg-28-111-2021](https://doi.org/10.5194/npg-28-111-2021)*

The original authors code can be found at: [zenodo.org/records/4354602](https://zenodo.org/records/4354602)

## Project Overview

This project implements a shallow water model (MSW) with data assimilation techniques, including Ensemble Kalman Filter (EnKF), Quadratic Programming Ensemble (QPEns), and a CNN for mass conservation corrections. It also includes uncertainty quantification via methods like Conformal Prediction.

### End-to-End Pipeline
<img width="801" height="1303" alt="full_project_pipeline" src="https://github.com/user-attachments/assets/4e4c1a4f-8b92-4e74-8a7c-d8f30b5bd87c" />

*High-level overview of the entire pipeline. (Click for full size; individual diagrams available in the next section.*

### Detailed Diagrams
<details>
<summary>Expand for Individual Diagrams</summary>
   <img width="766" height="1483" alt="msw_model" src="https://github.com/user-attachments/assets/1cd986e0-8458-4dfe-a76f-e55145c89f88" />
   <img width="795" height="1322" alt="data_generation" src="https://github.com/user-attachments/assets/c4908476-d52e-454f-8f41-0eeddc074a91" />
   <img width="565" height="950" alt="training" src="https://github.com/user-attachments/assets/b18aa6ac-5184-442d-9620-47da6f429dd1" />
   <img width="757" height="1548" alt="inference" src="https://github.com/user-attachments/assets/062225a6-244a-4310-abea-6501e0732774" />
   <img width="1238" height="1685" alt="history_data_generation" src="https://github.com/user-attachments/assets/549aef50-d6a7-4ea4-b91f-4f8b20fba7bf" />
   <img width="571" height="1028" alt="uncertainty_quantification" src="https://github.com/user-attachments/assets/17b4235e-6717-4874-bcc6-c905339f1d47" />
</details>

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