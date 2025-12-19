# RouteNet Project

This repository contains implementations and tools for network performance evaluation using Graph Neural Networks (GNNs), specifically RouteNet-Erlang and RouteNet-Fermi models. It includes prediction scripts for evaluating network metrics like delay, jitter, and loss under various traffic models and scheduling policies.

## Overview

The project is organized as follows:
- `RouteNet-Erlang/`: Implementation of RouteNet-Erlang for modeling networks with Erlang traffic distributions
- `RouteNet-Fermi/`: Implementation of RouteNet-Fermi for network modeling with Fermi-Dirac statistics
- `routenet/`: Conda environment and dependencies
- Prediction scripts for batch evaluation
- Analysis tools for processing results

## Setup

### Prerequisites

- Python 3.9
- Conda or Miniconda
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/mohammed-alaa40123/Route-Scout
cd Routenet
```

### 2. Set up the Environment

The project uses a conda environment located in `routenet/`. To activate it:

```bash
uv venv -n routenet --python=3.9
source routenet/bin/activate
```


### 3. Install Dependencies

For RouteNet-Erlang:
```bash
cd RouteNet-Erlang
uv pip install -r requirements.txt
```

For RouteNet-Fermi:
```bash
cd RouteNet-Fermi
uv pip install -r requirements.txt
```

### 4. Download Data

The models require network topology and traffic data. The data is typically stored in the respective model directories:

- For Erlang: `RouteNet-Erlang/data/`
- For Fermi: `RouteNet-Fermi/data/`

To download the datasets:

1. For RouteNet-Erlang, follow the instructions in `RouteNet-Erlang/data/README.md` (if available) or use the provided data generation scripts.

2. For RouteNet-Fermi, the data is included or can be generated using scripts in `RouteNet-Fermi/data/`.

The prediction scripts expect data in specific formats. Ensure the following directories exist:
- `RouteNet-Erlang/data/scheduling/`
- `RouteNet-Erlang/data/traffic_models/`
- `RouteNet-Fermi/data/traffic_models/`

## Prediction Scripts

The repository includes several bash scripts for running predictions across multiple samples and configurations.

### predict_with_candidate_routes_Traffic_models.sh

This script evaluates traffic models using candidate routes.

**Usage:**
```bash
./predict_with_candidate_routes_Traffic_models.sh [start_sample] [end_sample]
```

**Parameters:**
- `start_sample`: Starting sample index (default: 0)
- `end_sample`: Ending sample index (default: 0)

**What it does:**
- Generates candidate routes using k-shortest paths (k=5)
- Runs predictions for delay and jitter metrics
- Uses Fermi framework
- Processes train/test splits for gbn topology
- Outputs results to `Results/Fermi__TM_Samples/`

**Example:**
```bash
./predict_with_candidate_routes_Traffic_models.sh 20 60
```

This runs samples 20 through 60, creating output files like:
`TrafficModels_results_0_5_all_multiplexed_delay_TrafficModels_gbn_fermi_20.txt`

### predict_with_candidate_routes_scheduling.sh

Similar to the traffic models script but for scheduling policies.

**Usage:**
```bash
./predict_with_candidate_routes_scheduling.sh
```

**What it does:**
- Evaluates scheduling policies (wfq, wfq-drr-sp)
- Tests delay, jitter, and loss metrics
- Uses both Erlang and Fermi frameworks
- Processes multiple topologies (geant2, nsfnet, rediris)
- Outputs to `Results/erlang_S{SAMPLE_INDEX}/` and `Results/fermi_S{SAMPLE_INDEX}/`

### Other Scripts

- `run_samples_jobs.sh`: Batch job submission for HPC
- `run_samples_multi.sh`: Multi-sample processing
- `sample_job.sbatch`: SLURM job template

## Analysis Tools

### extract_candidate_best_counts.py

Extracts and analyzes prediction results.

**Usage:**
```bash
python extract_candidate_best_counts.py --results-root Results --out analysis_results/extract
```

**What it does:**
- Parses prediction output files
- Identifies best-performing routes (marked with *)
- Generates summary statistics
- Creates analysis reports

### analysis_scheduling_results.py

Analyzes scheduling prediction results.

## File Structure

```
Routenet/
├── RouteNet-Erlang/          # Erlang model implementation
├── RouteNet-Fermi/           # Fermi model implementation
├── routenet/                 # Conda environment
├── Results/                  # Prediction outputs
├── analysis_results/         # Analysis outputs
├── predict_*.sh              # Prediction scripts
├── extract_*.py              # Analysis scripts
├── k_shortest_routes.py      # Route generation
└── candidate_routes_*.txt    # Pre-computed routes
```

## Google Colab Setup

If running on Google Colab, follow these steps to set up the environment:

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install Miniconda:
```bash
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
!bash miniconda.sh -b -p /opt/conda
!rm miniconda.sh
```

3. Accept conda terms and create environment:
```bash
!source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!source /opt/conda/etc/profile.d/conda.sh && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
!source /opt/conda/etc/profile.d/conda.sh && conda create -n tf_py37 python=3.9 -y
!source /opt/conda/etc/profile.d/conda.sh && conda activate tf_py37
```

4. Install TensorFlow:
```bash
!/opt/conda/envs/tf_py37/bin/pip install tensorflow==2.6.*
```

5. Navigate to your project directory:
```python
%cd /content/drive/MyDrive/Routenet
```

6. Clone the repositories:
```bash
!git clone https://github.com/BNN-UPC/RouteNet-Fermi.git
!git clone https://github.com/BNN-UPC/RouteNet-Erlang.git
```

7. Install dependencies:
```bash
%cd /content/drive/MyDrive/Routenet/RouteNet-Fermi/
!/opt/conda/envs/tf_py37/bin/pip install -r requirements.txt
```

8. Download datasets:
```bash
%cd /content/drive/MyDrive/Routenet
!wget -O traffic_models.zip https://bnn.upc.edu/download/dataset-v6-traffic-models/
!wget -O scheduling.zip https://bnn.upc.edu/download/dataset-v6-scheduling/
!unzip traffic_models.zip -d data
!unzip scheduling.zip -d data/
!mv data RouteNet-Fermi/
```

Note: The notebook `K_shortest_path_Routenet (1).ipynb` contains the complete setup and prediction code for Colab environments.

**RouteNet-Erlang:**
```bibtex
@article{ferriol2022routenet,
  title={RouteNet-Erlang: A Graph Neural Network for Network Performance Evaluation},
  author={Ferriol-Galm{\'e}s, Miquel and Rusek, Krzysztof and Su{\'a}rez-Varela, Jos{\'e} and Xiao, Shihan and Cheng, Xiangle and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  journal={arXiv preprint arXiv:2202.13956},
  year={2022}
}
```

**RouteNet-Fermi:**
```bibtex
@article{ferriol2022routenet,
  title={RouteNet-Fermi: Network Modeling with Graph Neural Networks},
  author={Ferriol-Galm{\'e}s, Miquel and Paillisse, Jordi and Su{\'a}rez-Varela, Jos{\'e} and Rusek, Krzysztof and Xiao, Shihan and Shi, Xiang and Cheng, Xiangle and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  journal={arXiv preprint arXiv:2212.12070},
  year={2022}
}
```

**RouteScout:**
```bibtex
@article{ahmed2025routescout,
  title={RouteScout: Metric-Aware Path Picking over k-Shortest Routes},
  author={Ahmed, Mohamed and Tadros, Myron and Warsy, Bola},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE files in the respective RouteNet directories for details.

## Contributing

Please read CONTRIBUTING.md in the respective model directories for guidelines on contributing to this project.