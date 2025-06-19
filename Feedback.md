In the report Full-Field Trajectory Prediction in the NFL Using LSTM and Transformer Models, the task of predicting each player’s trajectory on the field, given past time series of player and ball positions, is tackled using RNN and Transformer models.

The abstract provides some background on the prediction problem, summarises methodology and results. The introduction offers basic relevant information on American football. However, more details could have been provided on the intended application of the predictions and the most interesting forecasting horizons. Related work covers trajectory forecasting in different contexts, including pedestrian movement and neural networks, but could more clearly articulate the closest related work or clarify if none exists.

The methodology uses RNN (LSTM) and Transformer neural networks with standard training methods (Adam optimiser, early stopping). Section 4 discusses autoregressive models and their implementation with RNN and Transformer. This section is basic and somewhat confusing: the autoregressive formulation for a specific player p_i is unclear regarding parameter sharing across players. The transformer part defines X(t) as a vector and presents a vector-valued recursion. The information presented in Section 4.2 might not be included. Section 5 introduces two different LSTM architectures — one for 1-step and one for 40-step prediction — which differ in architectural blocks without discussion. A unified model with user-definable prediction horizons could have been proposed.

Implementation involves processing a large dataset and training/evaluating RNN and Transformer models. The code spans many Python files and notebooks, indicating substantial coding and experimentation. Some notebooks contain error outputs (e.g., Big_transformer.ipynb). The codebase could have been consolidated and presented more clearly.

The dataset, from the NFL 2025 Big Data Bowl Kaggle competition, is well suited and described in Section 3.

Numerical evaluation uses MSE and average displacement loss (similar but differing units), introduced in Section 7.2. Metrics could have been defined clearly in one place. Training and validation loss curves over epochs are shown, but a summary table of MSE values across models and horizons would improve clarity. More discussion interpreting an average positional deviation of 1–2 yards per player relative to typical player displacement and field size would help readers assess prediction quality.

The conclusion summarises the problem and best accuracy achieved, followed by limitations and future work suggestions, including higher computational resource models and use of additional features.

Presentation quality is generally good, but structure and writing clarity could improve. Some plots have inconsistent labelling (e.g., Figures 1 and 2 versus 3 and 4).

**Mark: 68**
