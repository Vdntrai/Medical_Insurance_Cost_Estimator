# Medical Insurance Cost Predictor Using Linear Regression

This project aims to build a machine learning model to predict the cost of medical insurance based on various factors using Linear Regression. By leveraging data preprocessing techniques and Linear Regression, the project provides an effective tool for estimating insurance costs.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction
The Medical Insurance Cost Predictor uses Linear Regression to analyze various features such as age, BMI, smoking status, and more to predict the cost of medical insurance. This tool is designed to assist insurance companies, analysts, and individuals in estimating insurance premiums.

## Features
### Data Preprocessing:
- Handling missing values
- Encoding categorical features
- Feature scaling

### Model Training:
- Splitting the dataset into training and testing sets
- Training a Linear Regression model
- Evaluating model performance using metrics such as Mean Absolute Error (MAE) and R-squared

### Prediction System:
- Input features for cost estimation
- Predicting the insurance cost based on input features

## Installation
To get started with this project, follow the steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Vdntrai/Medical_Insurance_Cost_Estimator.git
    cd Medical_Insurance_Cost_Estimator
    ```


## Usage
To use the Medical Insurance Cost Predictor, follow these steps:

1. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Medical_Insurance_Cost_Predictor.ipynb
    ```

2. Import relevant dependencies:
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    ```

3. Train the model:
    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

4. Evaluate model performance:
    ```python
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    print(f'Train Mean Absolute Error: {train_mae:.2f}')

    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    print(f'Test Mean Absolute Error: {test_mae:.2f}')
    ```

5. Make predictions:
    ```python
    X_new = X_test[1234].reshape(1, -1)
    pred = model.predict(X_new)
    print(f'Predicted Insurance Cost: ${pred[0]:.2f}')
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
