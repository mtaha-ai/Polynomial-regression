# Polynomial Regression

This repository demonstrates the implementation of Polynomial Regression using stochastic gradient descent (SGD). The model predicts satellite velocity based on altitude. The dataset includes satellite altitude and velocity values, normalized using Min-Max scaling after being loaded.

## Steps

1. **Normalization of Data**

   To ensure numerical stability during training, Min-Max scaling is applied to both input features (altitude) and target values (velocity):

   ```
   x_scaled = (x - x_min) / (x_max - x_min)
   ```

   This maps the data to the range [0, 1].

2. **Polynomial Feature Expansion**

   The input feature X is expanded into a polynomial feature matrix \(X_poly\):

   ```
   X_poly = [1, x, x^2, ..., x^d]
   ```

   Here, \(d\) is the degree of the polynomial. This step allows the model to capture non-linear relationships.

3. **Model Representation**

   The polynomial regression model is expressed as:

   ```
   Y_pred = X_poly . Coefficients
   ```

   Where:
   - \(Y_pred\): Predicted values, calculated as the dot product of \(X_poly\) and the \(Coefficients\) matrix.
   - \(X_poly\): Polynomial feature matrix.
   - \(Coefficients\): A single matrix that represents both the weights \(W\) and bias \(b\) combined.

4. **Error Metric (MSE)**

   The loss function used is Mean Squared Error (MSE):

   ```
   MSE = (1 / n) * Σ(y_i - Y_pred_i)^2
   ```

   Here, \(Y_pred_i\) represents the predicted value for \(x_i\).

5. **Gradient Descent Optimization**

   The gradient of the MSE with respect to the \(Coefficients\) matrix is calculated as:

   ```
   Gradient = -(2 / n) * X_poly^T . (Y - Y_pred)
   ```

   The \(Coefficients\) matrix is updated iteratively:

   ```
   Coefficients = Coefficients - alpha * Gradient
   ```

   Where:
   - \(alpha\): Learning rate, controlling the step size.

   The process stops when:
   - The gradient norm is below a predefined threshold.
   - The maximum number of iterations is reached.

6. **Convergence Analysis**

   During training, the MSE at each iteration is recorded, producing a learning curve that illustrates model convergence.

7. **Model Prediction**

   Using the optimized \(Coefficients\), predictions for new inputs \(X\) are made by:

   ```
   Y_pred = X_poly . Coefficients
   ```

## Results and Visualization

1. **Best Fit Line**:
   - This is the best fit line created by the model. It demonstrates the relationship between altitude and velocity based on the trained polynomial regression model.

     ![Best Fit Line](Best%20fit%20line.png)

2. **Learning Curve**:
   - Illustrates the decrease in MSE over iterations, confirming the model's convergence.

     ![Learning Curve](learning%20curve.png)

## Conclusion

This project showcases the implementation of Polynomial Regression with stochastic gradient descent. By expanding input features into polynomial terms and optimizing coefficients iteratively, the model captures non-linear relationships effectively. The learning curve and best fit line validate the model's performance.