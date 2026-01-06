# multi-lead-ecg-reconstruction

MATLAB code for Deep Learning-based multi-lead ECG reconstruction



Files

* MultiInputFeature.m: Training script. Loads data, builds the model, trains it, and saves the trained network.
* eval\_MFI.m: Inference script. Uses the trained model to generate predicted ECG, compute STD (Monte Carlo dropout), and calculate errors.
* MakeHeatMap.m: Visualization script. Creates scatter plots (STD vs error), power-law fits, and STD heatmaps overlaid with ECG.



Requirements

MATLAB with Deep Learning Toolbox.



Workflow

1. Run MultiInputFeature.m to train the model.
2. Run eval\_MFI.m to perform inference.
3. Run MakeHeatMap.m to visualize uncertainty and errors.



Citation

If you use this code in your research, please cite the following paper:

R. Nakanishi, “Deep Learning-Based Multi-Lead ECG Reconstruction with Uncertainty Estimation,” Sensors, vol. 26, no. 1, 212, 2026. https://doi.org/10.3390/s26010212

