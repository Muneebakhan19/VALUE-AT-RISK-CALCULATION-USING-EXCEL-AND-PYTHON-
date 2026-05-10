# VALUE-AT-RISK-CALCULATION-USING-EXCEL
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def calculate_portfolio_metrics(portfolio_value, weights, ann_returns, ann_std_devs, covariance, days, confidence_level):
    """
    Calculates portfolio annual return, standard deviation, and Value at Risk (VaR) based on given inputs.

    Args:
        portfolio_value (float): The total value of the portfolio.
        weights (np.array): Array of weights for each asset in the portfolio.
        ann_returns (np.array): Array of annual returns for each asset.
        ann_std_devs (np.array): Array of annual standard deviations for each asset.
        covariance (float): The covariance between the two assets.
        days (int): Number of days for the investment horizon (e.g., 252 for annual trading days).
        confidence_level (float): The confidence level for VaR calculation (e.g., 0.99 for 99%).

    Returns:
        tuple: (port_ann_return, port_std_dev, cov_matrix, port_variance, var_1day)
    """
    # 1. Calculate Portfolio Annual Return
    port_ann_return = np.sum(ann_returns * weights)

    # 2. Create Covariance Matrix and Portfolio Std Dev
    cov_matrix = np.array([[ann_std_devs[0]**2, covariance],
                           [covariance, ann_std_devs[1]**2]])
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std_dev = np.sqrt(port_variance)

    # 3. Calculate VaR (Parametric Method)
    # Z-score for the given confidence level
    z_score = norm.ppf(1 - confidence_level)

    # Daily portfolio standard deviation (assuming 252 trading days in a year)
    port_daily_std_dev = port_std_dev / np.sqrt(252)

    # Daily portfolio return (assuming 252 trading days in a year)
    port_daily_return = port_ann_return / 252

    # 1-day VaR
    var_1day = portfolio_value * (port_daily_return + z_score * port_daily_std_dev)

    print(f"Portfolio Annual Return: {port_ann_return:.2%}")
    print(f"Portfolio Annual Std Dev: {port_std_dev:.4%}")
    print(f"{int(confidence_level*100)}% 1-Day Value at Risk (VaR): ${var_1day:,.2f}")

    return port_ann_return, port_std_dev, cov_matrix, port_variance, var_1day

# Portfolio Inputs - These can now be entered interactively by the user
portfolio_value = float(input("Enter the total portfolio value (e.g., 200000): "))
weights_input = input("Enter weights for each asset, comma-separated (e.g., 0.6,0.4): ")
weights = np.array([float(w) for w in weights_input.split(',')])

ann_returns_input = input("Enter annual returns for each asset, comma-separated (e.g., 0.1746,0.3692): ")
ann_returns = np.array([float(r) for r in ann_returns_input.split(',')])

ann_std_devs_input = input("Enter annual standard deviations for each asset, comma-separated (e.g., 0.243,0.632): ")
ann_std_devs = np.array([float(s) for s in ann_std_devs_input.split(',')])

covariance = float(input("Enter the covariance between the two assets (e.g., 0.07189): "))
days = int(input("Enter the number of days for the investment horizon (e.g., 50): "))
confidence_level = float(input("Enter the confidence level for VaR calculation (e.g., 0.99): "))

# Call the function to calculate and get the results
port_ann_return, port_std_dev, cov_matrix, port_variance, var_1day = calculate_portfolio_metrics(
    portfolio_value, weights, ann_returns, ann_std_devs, covariance, days, confidence_level
)
