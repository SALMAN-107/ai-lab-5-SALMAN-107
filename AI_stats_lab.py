"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Add bias column
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # Initialize parameters
    theta = np.zeros(X_train.shape[1])
    
    # Gradient Descent
    n = X_train.shape[0]
    for epoch in range(epochs):
        predictions = X_train @ theta
        error = predictions - y_train
        mse_gradient = (2/n) * (X_train.T @ error)
        l1_grad = lambda_reg * np.sign(theta)
        l1_grad[0] = 0  # Do not regularize the bias term
        total_gradient = mse_gradient + l1_grad
        theta = theta - lr * total_gradient
    
    # Compute predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta
    
    # Compute metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    
    # Use BMI feature only
    X = X[:, 2].reshape(-1, 1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prepare empty lists
    degrees = []
    train_mse = []
    test_mse = []
    
    # Loop over polynomial degrees
    for d in range(1, max_degree + 1):
        # Create polynomial feature matrix
        X_poly = np.hstack([X_train ** i for i in range(d+1)])
        X_test_poly = np.hstack([X_test ** i for i in range(d+1)])
        
        # Train model using Normal Equation
        theta = np.linalg.pinv(X_poly.T @ X_poly) @ X_poly.T @ y_train
        
        # Predictions
        train_pred = X_poly @ theta
        test_pred = X_test_poly @ theta
        
        # Compute errors
        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)
        
        # Store results
        degrees.append(d)
        train_mse.append(train_error)
        test_mse.append(test_error)
    
    return {
        "degrees": degrees,
        "train_mse": train_mse,
        "test_mse": test_mse
    }
