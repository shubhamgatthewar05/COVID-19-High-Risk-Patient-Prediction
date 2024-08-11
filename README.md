
# COVID-19 High-Risk Patient Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Advanced Techniques](#advanced-techniques)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
This project predicts whether a COVID-19 patient is at high risk using a machine learning model. The project involves data preprocessing, feature engineering, model selection, training, and evaluation.

## Features
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling features.
- **Modeling:** Training a Random Forest classifier with hyperparameter tuning and SMOTE for imbalanced data.
- **Evaluation:** Assessing model performance with accuracy, confusion matrix, and classification report.

## Dataset
The dataset used in this project contains patient-related features such as `SEX`, `AGE`, `PREGNANT`, `DIABETES`, etc., with the target variable being `CLASIFFICATION_FINAL`, which indicates the risk level of the patient.

## Requirements
- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/covid19-high-risk-prediction.git
    cd covid19-high-risk-prediction
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Place the dataset in the root directory or update the `path_to_your_dataset.csv` in the code.

## Usage
1. Run the script to preprocess data, train the model, and evaluate results:
    ```bash
    python model_training.py
    ```
2. The script will output the model's accuracy, confusion matrix, and classification report, along with a feature importance plot.

## Modeling Process
- **Data Preprocessing:**
  - Categorical features were label encoded.
  - Missing values were imputed with median or most frequent values.
  - SMOTE was applied to balance the classes.
- **Model Training:**
  - A Random Forest classifier was used for its robustness.
  - Hyperparameter tuning was conducted using GridSearchCV.
- **Feature Scaling:**
  - StandardScaler was used to standardize the features.
- **Evaluation:**
  - Accuracy, confusion matrix, and classification report were generated.
  - Feature importance was visualized.

## Results
- The final model achieved an accuracy of `94%` on the test set.
- The confusion matrix and classification report provided insights into the model's performance across different classes.

## Advanced Techniques
- **Hyperparameter Tuning:** GridSearchCV was used to find the optimal parameters for the Random Forest model.
- **Ensemble Learning:** A Voting Classifier was suggested for combining multiple models to improve prediction accuracy.

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.
