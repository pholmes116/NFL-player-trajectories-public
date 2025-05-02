---

### 📘 README.md

```markdown
# Shallow Learning Projects – NFL Big Data Bowl 2025

This repository contains tools, code examples, and documentation for developing shallow learning models, with a specific focus on the NFL Big Data Bowl 2025 dataset.

## 🏈 Overview

This project aims to predict player trajectories using data from the 2025 Big Data Bowl competition. It includes:

- Data loading utilities  
- Visualization tools  
- Baseline model implementations  
- Notebooks for exploration and experimentation  
- Instructions for environment setup  

## 📁 Project Structure

```
.\
├── code/\
│   ├── Dataloader_maker.py\
│   ├── Polars_pipeline.py\
│   ├── custom_visualizer.py\
│   ├── download_nfl_bigdatabowl2025_data.py\
│   ├── explore_dataset.ipynb\
│   ├── Models/\
│   │   ├── Base_transformer.py\
│   │   ├── Base_transformer.ipynb\
│   │   └── Base_transformer_trainer.py\
│   └── ...\
├── environment_instructions.md\
├── nfl_env.yml\
├── PROPOSAL.md\
├── ST456-project-marking.pdf\
└── README.md\
```

## ⚙️ Setup Instructions

1. **Install Conda Environment**  
   Create the environment using the provided YAML file:

   ```bash
   conda env create -f nfl_env.yml
   conda activate nfl2025
   ```

2. **Download the Dataset**
   Run the following script to download the Big Data Bowl 2025 dataset:

   ```bash
   python code/download_nfl_bigdatabowl2025_data.py
   ```

3. **Explore the Dataset**
   Use the Jupyter notebooks in the `code/` directory to begin exploration:

   * `explore_dataset.ipynb`
   * `test_custom_visualizer.ipynb`

4. **Run the Polars Pipeline**
   Use the following scripts to preprocess and load data:

   * `Polars_pipeline.py`
   * `Dataloader_maker.py`

## 🧠 Models

### LSTM Models

* One-step prediction: `code/Models/LSTM1.ipynb`
* Forty-step prediction: `code/Models/LSTM40.ipynb`

### Transformer Models

* Base implementation: `code/Models/Base_transformer.ipynb`
* Larger architecture: `code/Models/Big_transformer.ipynb`
* Physics-informed loss functions (partial): `code/Models/Base_transformer_physics.ipynb`

## 📄 Additional Files

* `PROPOSAL.md`: Project proposal
* `ST456-project-marking.pdf`: Marking criteria (for coursework)
* `environment_instructions.md`: Step-by-step setup guide
* `.gitignore`: Git configuration

---

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)

```
