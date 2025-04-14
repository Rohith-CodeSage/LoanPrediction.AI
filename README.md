
# LoanPrediction.AI

LoanPrediction.AI is an AI-powered solution for predicting personal loan approvals using an Artificial Neural Network (ANN). The project preprocesses a custom dataset, applies a boosting logic to improve approval predictions, and utilizes TensorFlow/Keras for model training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The goal of LoanPrediction.AI is to streamline the loan approval process by predicting whether a loan application will be approved or not. The model is designed to incorporate not only the raw features (like age, credit score, income, etc.) but also a custom boosting logic that adjusts approvals based on certain key applicant criteria.

## Features

- **Custom Data Preprocessing:** Applies a boosting function to raise approval chances based on criteria such as age, credit score, employment, and more.
- **ANN Model:** Built with TensorFlow/Keras, the ANN has multiple hidden layers to capture complex patterns in data.
- **Visualizations:** Generates and saves plots for training accuracy & loss, confusion matrix, and ROC curve.
- **Persistence:** Saves the trained model, scaler, and training history for later use or further analysis.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rohith-CodeSage/LoanPrediction.AI.git
   cd LoanPrediction.AI
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *If a `requirements.txt` file is not present, install the following packages manually:*
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - tensorflow
   - joblib

## Usage

1. **Dataset Location:**  
   Ensure that the dataset is located at `loan_app/ml_model/personal_loan_dataset_updated.csv`.

2. **Run the script:**  
   Execute the training script (e.g., `train_model.py` if that’s the filename) with:
   ```bash
   python train_model.py
   ```
   This script will:
   - Load and preprocess the data.
   - Train the ANN with the provided boosting logic.
   - Evaluate model performance.
   - Save the model, scaler, and performance plots.

3. **Outputs:**  
   - **Model & Scaler:** Stored under `loan_app/ml_model/`.
   - **Plots:** Saved in `loan_app/static/loan_app/plots/` (includes accuracy/loss, ROC curve, confusion matrix, and training history CSV).

## Project Structure

```
LoanPrediction.AI/
├── loan_app/
│   ├── ml_model/
│   │   ├── personal_loan_dataset_updated.csv
│   │   ├── loan_ann_model.h5          # Saved model
│   │   └── scaler.pkl                 # Saved scaler
│   ├── static/
│   │   └── loan_app/
│   │       └── plots/
│   │           ├── ann_loss_accuracy.png
│   │           ├── confusion_matrix.png
│   │           ├── roc_curve.png
│   │           └── training_history.csv
├── README.md
└── requirements.txt                   # List of dependencies
```

## Model Details

- **Architecture:**
  - Input Layer: Matches the number of features.
  - Hidden Layers: 64 neurons → 32 neurons → 16 neurons, each with ReLU activation.
  - Output Layer: Single neuron with sigmoid activation for binary classification.

- **Compilation Settings:**
  - Optimizer: Adam
  - Loss: Binary Crossentropy
  - Metric: Accuracy

- **Custom Approval Boost Logic:**  
  A function increases the approval likelihood for applicants who are within a specific age range, have high credit scores, are employed, own their residence, and maintain a balanced loan-to-income ratio.

## Evaluation

After training, the script evaluates the model by:
- Plotting training/validation accuracy and loss over epochs.
- Generating a confusion matrix and ROC curve (with AUC).
- Printing a detailed classification report.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, suggestions, or collaboration, feel free to reach out:
- **GitHub:** [Rohith-CodeSage](https://github.com/Rohith-CodeSage)
```

---

I hope this README helps set up everything clearly. Let me know if you need any more tweaks or details—I've got your back!
