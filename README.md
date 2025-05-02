```

### 📘 README.md

```markdown
# Shallow Learning Projects – NFL Big Data Bowl 2025

This repository contains tools, code examples, and documentation for developing shallow learning models, with a specific focus on the NFL Big Data Bowl 2025 dataset.

## 🏈 Overview

This project aims to predict player trajectory using data from the 2025 Big Data Bowl competition:
- Data loading utilities
- Visualization tools
- Baseline model implementations
- Notebooks for exploration and experimentation
- Instructions for environment setup

## 📁 Project Structure

```

.
├── code/
│   ├── Dataloader\_maker.py
│   ├── Polars\_pipeline.py
│   ├── custom\_visualizer.py
│   ├── download\_nfl\_bigdatabowl2025\_data.py
│   ├── explore\_dataset.ipynb
│   ├── Models/
│   │   ├── Base\_transformer.py
│   │   ├── Base\_transformer.ipynb
│   │   └── Base\_transformer\_trainer.py
│   └── ...
├── environment\_instructions.md
├── nfl\_env.yml
├── PROPOSAL.md
├── ST456-project-marking.pdf
└── README.md

````

## ⚙️ Setup Instructions

1. **Install Conda Environment**  
   Create the environment using the provided YAML file:

   ```bash
   conda env create -f nfl_env.yml
   conda activate nfl2025
````

2. **Download Dataset**
   Run the following script to download the Big Data Bowl 2025 dataset:

   ```bash
   python code/download_nfl_bigdatabowl2025_data.py
   ```

3. **Explore the Dataset**
   Use the Jupyter notebooks in the `code/` directory to start exploring the data:

   * `explore_dataset.ipynb`
   * `test_custom_visualizer.ipynb`

4. **Run the Polars Pipeline**
   Use the scripts in the `code/` directory to start exploring the data:

   * `Polars_pipelien.py`
   * `Dataloader_maker.py`
   

## 🧠 Models

LSTM and 
  * See `code/Models/LSTM1.ipynb` for the one step implementation.
  * See `code/Models/LSTM40.ipynb` for the fourty step implementation.
Transformer Models:
  * See `code/Models/Base_transformer.ipynb` for the base implementation.
  * See `code/Models/Big_transformer.ipynb` for the big implementation.
  * See `code/Models/Base_transformer_physics.ipynb` to see some loss functions we ran out of time to implement.

## 📄 Additional Files

* `PROPOSAL.md`: Project proposal template
* `ST456-project-marking.pdf`: Marking criteria (likely for coursework)
* `environment_instructions.md`: Step-by-step setup guide
* `.gitignore`: Git configuration
```
