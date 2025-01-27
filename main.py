from model import PolynomialRegressor
from utils import Scaler, Plot
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('data.csv')
    X = df['Altitude (km)'].values
    y = df['Velocity (km/s)'].values

    # Scale the data
    scaler = Scaler()
    X_scaled = scaler.MinMaxScaler(X)
    y_scaled = scaler.MinMaxScaler(y)

    # Fit the model
    model = PolynomialRegressor(learning_rate=0.1, degree=3, max_iter=1000, tolerance=1e-3)
    model.fit(X_scaled, y_scaled)

    # Predict and plot
    X_smooth = np.linspace(X_scaled.min(), X_scaled.max(), 100)
    y_pred = model.predict(X_smooth)

    Plot.plot_best_fit(X_scaled, y_scaled, X_smooth, y_pred)
    Plot.plot_learning_curve(model.errors_history)

if __name__ == '__main__':
    main()