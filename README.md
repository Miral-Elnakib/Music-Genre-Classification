# Music-Genre-Classification
Music Genre Classification with CNN-LSTM Hybrid Model

Music Genre Classification with CNN-LSTM Hybrid Model
   

Welcome to the Music Genre Classification project! This repository implements a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to classify music genres from audio data. Trained on the GTZAN dataset, this model processes audio spectrograms to predict genres such as Blues, Classical, Rock, and more with high accuracy.

🎵 Project Overview
This project leverages the power of deep learning to analyze audio time-series data and classify music into 10 distinct genres. The pipeline includes audio preprocessing (splitting songs into 3-second clips, converting to STFT spectrograms), a hybrid CNN-LSTM architecture, and model training with validation. Whether you're a machine learning enthusiast, a music lover, or a data scientist, this project offers a robust framework to explore audio classification.

Key Features
Audio Preprocessing: Converts raw audio files into Short-Time Fourier Transform (STFT) spectrograms.
Hybrid Model: Combines CNN for feature extraction and Bidirectional LSTM for temporal sequence modeling.
High Accuracy: Achieves strong performance on the GTZAN dataset (see training results below).
Modular Code: Well-structured Jupyter notebooks for preprocessing and model training.
📊 Results
The model was trained for 70 epochs with the following performance:

Training Accuracy: ~85%+
Validation Accuracy: Peaks at ~81% (see plot below).
Loss: Converges effectively, with the best validation loss at 0.10018.


Accuracy over epochs for training and validation sets.
