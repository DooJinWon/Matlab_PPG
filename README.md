This repository contains a MATLAB project for predicting the mean Arterial Blood Pressure (ABP) from Photoplethysmogram (PPG) signals. It covers the full pipeline, from data preprocessing to model training, and finally converting the predicted results into real blood pressure units (mmHg).

In the data preprocessing stage, raw PPG and ABP signals are segmented into fixed-length windows. For each window, three input channels are created:  
1) The original PPG signal with z-score normalization applied,  
2) The first derivative of the PPG, and  
3) The second derivative of the PPG.  
The label for each window is defined as the mean value of the first channel (PPG). The dataset is then split into training and testing sets.

Two prediction model variants are provided. The first is based on Long Short-Term Memory (LSTM) networks, which are well-suited for learning long-term dependencies in sequential data. The second uses Gated Recurrent Units (GRU), which have a simpler architecture and require fewer computations, allowing for faster training. Both models take the 3-channel PPG sequence as input and predict the mean ABP using the features extracted from the final time step. Training is performed with the Adam optimizer and gradient clipping, and model performance is monitored using a validation set.

Since the models output predictions in standardized units, a post-processing step converts these values into real blood pressure units (mmHg). This conversion uses the mean and standard deviation of the training targets, enabling clinically interpretable results. The converted predictions are evaluated using Root Mean Squared Error (RMSE) and Pearson’s correlation coefficient, and visualized with time-series plots and parity plots for easier interpretation.

This project is implemented using the MATLAB Deep Learning Toolbox and is intended for research and experimental purposes. It is not suitable for medical diagnosis. Users are encouraged to adjust parameters such as window length, stride, and the number of hidden units according to the characteristics of their dataset.
