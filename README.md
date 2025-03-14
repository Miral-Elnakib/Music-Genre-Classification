# Music-Genre-Classification
Music Genre Classification with CNN-LSTM Hybrid Model

   
# Music Genre Classification with CNN-LSTM Hybrid Model

Welcome to the Music Genre Classification project! This repository implements a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to classify music genres from audio data. Trained on the GTZAN dataset, this model processes audio spectrograms to predict genres such as Blues, Classical, Rock, and more with high accuracy.

## 🎵 Project Overview
This project leverages the power of deep learning to analyze audio time-series data and classify music into 10 distinct genres. The pipeline includes:
- **Audio preprocessing** (splitting songs into 3-second clips, converting to STFT spectrograms)
- **A hybrid CNN-LSTM architecture** for deep feature extraction and sequence modeling
- **Model training and validation** to achieve high accuracy

Whether you're a machine learning enthusiast, a music lover, or a data scientist, this project offers a robust framework to explore audio classification.

## 🚀 Key Features
- **Audio Preprocessing:** Converts raw audio files into Short-Time Fourier Transform (STFT) spectrograms.
- **Hybrid Model:** Combines CNN for feature extraction and Bidirectional LSTM for temporal sequence modeling.
- **High Accuracy:** Achieves strong performance on the GTZAN dataset (see training results below).
- **Modular Code:** Well-structured Jupyter notebooks for preprocessing and model training.

## 📊 Results
The model was trained for **70 epochs** with the following performance:
- **Training Accuracy:** ~85%+
- **Validation Accuracy:** Peaks at ~81%
- **Loss:** Converges effectively, with the best validation loss at **0.10018**

### 📈 Accuracy over epochs for training and validation sets
(Include accuracy/loss plots here)

## 📂 Project Structure
```
📁 Music-Genre
│-- 📂 data/        # Dataset and preprocessed features
│-- 📂 models/      # Trained models and saved weights
│-- 📂 notebooks/   # Jupyter notebooks for preprocessing and training
│-- 📂 src/         # Source code for feature extraction, training, and evaluation
│-- README.md       # Project documentation
│-- requirements.txt # Dependencies
│-- train.py        # Training script
│-- predict.py      # Inference script
```

## 🛠️ Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Music-Genre.git
   cd Music-Genre
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🎯 Usage
- **Train the Model:**
  ```sh
  python train.py
  ```
- **Make Predictions:**
  ```sh
  python predict.py --input path/to/audio/file.wav
  ```

## 🧠 Technologies Used
- Python
- Librosa (for audio feature extraction)
- TensorFlow / PyTorch (for deep learning models)
- Scikit-learn (for traditional machine learning models)

## ✨ Contributions
Feel free to fork this repository and submit pull requests if you’d like to contribute!

## 📜 License
This project is licensed under the MIT License.
