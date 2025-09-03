# Predicting player trajectories – NFL Big Data Bowl 2025

This repository contains tools, code examples, and documentation for developing shallow learning models, with a specific focus on the NFL Big Data Bowl 2025 dataset.

## Authors

Sebastian Ballesteros, Peter Holmes, Juan Bautista Sosa, Mufan Yang

## Overview

This project aims to predict player trajectories using data from the 2025 Big Data Bowl competition. It includes:

- Data loading utilities  
- Visualization tools  
- Baseline model implementations  
- Notebooks for exploration and experimentation  
- Instructions for environment setup  

## Project Structure

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
├── Report.pdf\
└── README.md

## Setup Instructions

1. **Install Conda Environment**  
   Create the environment using the provided YAML file:
   ```bash
   conda env create -f nfl_env.yml
   conda activate nfl2025
   ```

2. **Download the Dataset**
   ```bash
   python code/download_nfl_bigdatabowl2025_data.py
   ```

3. **Explore the Dataset**
   - `explore_dataset.ipynb`
   - `test_custom_visualizer.ipynb`

4. **Run the Polars Pipeline**
   - `Polars_pipeline.py`
   - `Dataloader_maker.py`

## Models

### LSTM Models
- One-step prediction: `code/Models/LSTM1.ipynb`
- Forty-step prediction: `code/Models/LSTM40.ipynb`

### Transformer Models
- Base implementation: `code/Models/Base_transformer.ipynb`
- Larger architecture: `code/Models/Big_transformer.ipynb`
- Physics-informed loss functions: `code/Models/Base_transformer_physics.ipynb`

## Additional Files
- `environment_instructions.md`: Setup guide
- `.gitignore`: Git configuration

## Contact

For questions or contributions, reach out via GitHub Issues or email pholmes116@gmail.com.
