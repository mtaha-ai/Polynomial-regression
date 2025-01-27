import matplotlib.pyplot as plt

class Scaler:
    def MinMaxScaler(self, data):
        """
        Scale the input data to a range between 0 and 1 using Min-Max Scaling.
        :param data: Input data, shape (n_samples,)
        :return: Scaled data, shape (n_samples,)
        """
        data_min = data.min()
        data_max = data.max()
        data_scaled = (data - data_min) / (data_max - data_min)
        return data_scaled

class Plot:
    @staticmethod
    def plot_best_fit(X, y, X_smooth, y_pred):
        """
        Plot the data points and the best fit line.
        """
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.plot(X_smooth, y_pred, color='red', label='Best Fit Line')
        plt.xlabel('Altitude')
        plt.ylabel('Velocity')
        plt.title('Best Fit Line')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_learning_curve(errors_history):
        """
        Plot the learning curve using the recorded errors history.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(errors_history, label='Training Error (MSE)', linewidth=2)
        plt.title('Learning Curve', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Mean Squared Error', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.show()