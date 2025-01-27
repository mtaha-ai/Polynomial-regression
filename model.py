import numpy as np

class PolynomialRegressor:
    def __init__(self, learning_rate=0.01, degree=2, max_iter=1000, tolerance=1e-3):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.coefficients = None
        self.degree = degree
        self.errors_history = []  # Track MSE during training

    def fit(self, X, y):
        """
        Fit the polynomial regression model using gradient descent.
        :param X: numpy array, shape (n_samples,)
        :param y: numpy array, shape (n_samples,)
        """
        # Expand X to include polynomial terms
        X_poly = self.expand_features(X)
        n_samples, n_features = X_poly.shape

        # Initialize coefficients
        self.coefficients = np.zeros(n_features)

        # Perform gradient descent
        for iteration in range(self.max_iter):
            # Compute predictions for all samples
            y_pred = X_poly @ self.coefficients

            # Compute errors for all samples
            errors = y_pred - y

            # Compute gradients
            gradient = (X_poly.T @ errors) / n_samples

            # Update parameters
            self.coefficients -= self.learning_rate * gradient

            # Record Mean Squared Error for the learning curve
            mse = np.mean(errors ** 2)
            self.errors_history.append(mse)

            # Debugging: Log progress and gradient norms
            if iteration % 10 == 0:
                gradient_norm = np.linalg.norm(gradient)
                print(f"Iteration {iteration}, MSE = {mse}, Gradient Norm = {gradient_norm}")

            # Convergence check
            if np.linalg.norm(gradient) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

    def predict(self, X):
        """
        Predict target values using the learned weights and bias.
        """
        X_poly = self.expand_features(X)
        return X_poly @ self.coefficients
    
    def expand_features(self, X):
        """
        Expand X into a matrix with polynomial terms.
        :param X: numpy array, shape (n_samples,)
        :return: X_poly, shape (n_samples, degree + 1)
        """
        X = X.reshape(-1, 1)
        X_poly = np.hstack([X**i for i in range(self.degree + 1)])
        return X_poly