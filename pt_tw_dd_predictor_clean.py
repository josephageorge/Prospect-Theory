import pandas as pd
import numpy as np
from functools import lru_cache
from tqdm import tqdm
from dataclasses import dataclass
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import sys
CACHE_SIZE = 10000
DATA_DIR = '.'
@dataclass(frozen=True)
class Parameters:
    a: float = 1.0
    b: float = 1.0
    g: float = 1.0
    l: float = 1.0
    tw: int = 68
    epsilon_gains: float = 0.0
    epsilon_losses: float = 0.0
def load_dataset() -> pd.DataFrame:
    try:
        data = pd.read_csv("/Users/joeygeorge/Desktop/PT_TW_DD PREDICTIONS.csv")
        print(f"Successfully loaded dataset from /Users/joeygeorge/Desktop/PT_TW_DD PREDICTIONS.csv")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
class PT_TW_DDModel:
    def __init__(self):
        self.data = None
        self.best_fits = {}
        self.convergence_data = {}
    @lru_cache(maxsize=CACHE_SIZE)
    def probability_weighting_function(self, probability: float, gamma: float) -> float:
        return probability ** gamma
    @lru_cache(maxsize=CACHE_SIZE)
    def gain(self, amount: int, price: int, j: int, a: float, g: float) -> float:
        price_diff = j - price
        if price_diff <= 0:
            return 0
        inner_bracket1 = amount * (price_diff) ** a
        return inner_bracket1 * self.probability_weighting_function(1 / j, g)
    @lru_cache(maxsize=CACHE_SIZE)
    def loss(self, amount: int, price: int, j: int, b: float, l: float, g: float) -> float:
        price_diff = price - j
        if price_diff <= 0:
            return 0
        inner_bracket2 = amount * (price_diff) ** b
        return inner_bracket2 * self.probability_weighting_function(1 / j, g) * l
    @lru_cache(maxsize=CACHE_SIZE)
    def delayed_discounting(self, time_diff: int, epsilon: float) -> float:
        if time_diff < 0:
            return 1.0
        return 1 / (1 + epsilon * time_diff)
    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_gain(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        ev = 0.0
        max_price = 15
        for future_day in range(day + 1, day + fit.tw + 1):
            time_diff = future_day - day
            discount_factor = self.delayed_discounting(time_diff, fit.epsilon_gains)
            for future_price in range(price + 1, max_price + 1):
                gain_value = self.gain(amount, price, future_price, fit.a, fit.g)
                ev += gain_value * discount_factor
        return ev
    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_loss(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        ev = 0.0
        for future_day in range(day + 1, day + fit.tw + 1):
            time_diff = future_day - day
            discount_factor = self.delayed_discounting(time_diff, fit.epsilon_losses)
            for future_price in range(1, price):
                loss_value = self.loss(amount, price, future_price, fit.b, fit.l, fit.g)
                ev += loss_value * discount_factor
        return ev
    def error_of_fit(self, subject: int, fit: Parameters) -> float:
        subject_data = self.data[self.data['Subject'] == subject]
        stored_cols = [col for col in self.data.columns if col.startswith('Stored')]
        sold_cols = [col for col in self.data.columns if col.startswith('Sold')]
        price_cols = [col for col in self.data.columns if col.startswith('Price')]
        stored = subject_data[stored_cols].values.flatten()
        sold = subject_data[sold_cols].values.flatten()
        prices = subject_data[price_cols].values.flatten()
        days = np.arange(1, len(stored) + 1)
        predicted_sold = self.predict_sales(subject, days, stored, prices, sold, fit)
        error = np.sum((sold - predicted_sold) ** 2)
        return error
    def predict_sales(self, subject: int, days: np.ndarray, stored: np.ndarray, prices: np.ndarray, actual_sold: np.ndarray, params: Parameters) -> np.ndarray:
        predicted_sold = np.zeros_like(stored)
        for idx, day in enumerate(days):
            amount = stored[idx]
            price = prices[idx]
            if amount == 0:
                predicted_sold[idx] = 0
                continue
            ev_gain = self.expected_value_gain(day, price, amount, params)
            ev_loss = self.expected_value_loss(day, price, amount, params)
            if ev_gain >= abs(ev_loss):
                predicted_sold[idx] = amount
            else:
                predicted_sold[idx] = 0
        return predicted_sold
    def fit_one_subject(self, subject: int, start_fit: Parameters) -> None:
        bounds = [(0.01, 1.0),
                  (0.01, 1.0),
                  (0.01, 3.0),
                  (1.0, 10.0),
                  (0.0, 1.0),
                  (0.0, 1.0)]
        convergence_history = []
        def objective(params):
            fit = Parameters(a=params[0], b=params[1], g=params[2], l=params[3], tw=start_fit.tw, epsilon_gains=params[4], epsilon_losses=params[5])
            error = self.error_of_fit(subject, fit)
            convergence_history.append(error)
            return error
        result = differential_evolution(objective, bounds)
        self.convergence_data[subject] = convergence_history
        best_fit = Parameters(a=result.x[0], b=result.x[1], g=result.x[2], l=result.x[3], tw=start_fit.tw, epsilon_gains=result.x[4], epsilon_losses=result.x[5])
        self.best_fits[subject] = best_fit
        print(f"\nSubject {subject} best-fit parameters: {best_fit}\n")
    def fit_all_subjects(self, start_fit: Parameters):
        for subject in self.data['Subject'].unique():
            print(f"Fitting subject {subject}")
            self.fit_one_subject(subject, start_fit)
        for subject, error_history in self.convergence_data.items():
            plt.figure(figsize=(8, 6))
            plt.plot(error_history, label=f'Subject {subject}')
            plt.title(f"Convergence of Differential Evolution for Subject {subject}")
            plt.xlabel("Iteration")
            plt.ylabel("Objective Function Value")
            plt.legend()
            plt.show()
def main(version: str):
    model = PT_TW_DDModel()
    data = load_dataset()
    model.data = data
    start_fit = Parameters(a=0.5, b=0.5, g=1.0, l=2.25, epsilon_gains=0.5, epsilon_losses=0.5)
    model.fit_all_subjects(start_fit)
if __name__ == '__main__':
    version = "tw_dd_v4"
    main(version=version)