import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Simulated Data
data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Maharashtra': [5000, 7000, 8500, 6000, 9500, 12000, 13500],
    'Karnataka': [4000, 6500, 8000, 5000, 8700, 11000, 12500]
}

# Create DataFrame
df = pd.DataFrame(data)

# Train linear regression models
maharashtra_model = LinearRegression()
maharashtra_model.fit(df['Year'].values.reshape(-1, 1), df['Maharashtra'].values.reshape(-1, 1))

karnataka_model = LinearRegression()
karnataka_model.fit(df['Year'].values.reshape(-1, 1), df['Karnataka'].values.reshape(-1, 1))

# Predict future years
future_years = np.array(range(2024, 2028)).reshape(-1, 1)

# Predict Maharashtra and Karnataka GST for future years
maharashtra_predicted = maharashtra_model.predict(future_years).flatten()
karnataka_predicted = karnataka_model.predict(future_years).flatten()

# Combine actual and predicted data for Monte Carlo simulation
years_combined = np.concatenate((df['Year'].values, future_years.flatten()))
maharashtra_combined = np.concatenate((df['Maharashtra'].values, maharashtra_predicted))
karnataka_combined = np.concatenate((df['Karnataka'].values, karnataka_predicted))

# Monte Carlo Simulation for ideal GST growth
num_simulations = 10000
maharashtra_ideal = []
karnataka_ideal = []

# Define growth rate ranges (e.g., ±10% around the predicted trend)
maharashtra_growth_rate_range = np.linspace(0.08, 0.12, num_simulations)  # 8%-12% annual growth
karnataka_growth_rate_range = np.linspace(0.07, 0.11, num_simulations)    # 7%-11% annual growth

# Run simulations
for _ in range(num_simulations):
    maharashtra_sim = [maharashtra_combined[0]]
    karnataka_sim = [karnataka_combined[0]]

    for i in range(1, len(years_combined)):
        maharashtra_sim.append(maharashtra_sim[-1] * (1 + np.random.choice(maharashtra_growth_rate_range)))
        karnataka_sim.append(karnataka_sim[-1] * (1 + np.random.choice(karnataka_growth_rate_range)))
    
    maharashtra_ideal.append(maharashtra_sim)
    karnataka_ideal.append(karnataka_sim)

# Calculate mean and confidence intervals
maharashtra_ideal_mean = np.mean(maharashtra_ideal, axis=0)
karnataka_ideal_mean = np.mean(karnataka_ideal, axis=0)
maharashtra_ideal_ci = np.percentile(maharashtra_ideal, [5, 95], axis=0)
karnataka_ideal_ci = np.percentile(karnataka_ideal, [5, 95], axis=0)

# Plotting the actual, predicted, and ideal trends
plt.figure(figsize=(14, 8))

# Plot actual and predicted
plt.plot(years_combined, maharashtra_combined, label='Maharashtra (Actual + Predicted)', marker='o', color='blue')
plt.plot(years_combined, karnataka_combined, label='Karnataka (Actual + Predicted)', marker='o', color='green')

# Plot ideal trends
plt.plot(years_combined, maharashtra_ideal_mean, label='Maharashtra Ideal (Monte Carlo Mean)', linestyle='--', color='cyan')
plt.fill_between(years_combined, maharashtra_ideal_ci[0], maharashtra_ideal_ci[1], color='cyan', alpha=0.2, label='Maharashtra 90% CI')

plt.plot(years_combined, karnataka_ideal_mean, label='Karnataka Ideal (Monte Carlo Mean)', linestyle='--', color='lime')
plt.fill_between(years_combined, karnataka_ideal_ci[0], karnataka_ideal_ci[1], color='lime', alpha=0.2, label='Karnataka 90% CI')

# Add labels, legend, and title
plt.xlabel('Year')
plt.ylabel('GST Collection')
plt.title('GST Collection Trends with Monte Carlo Simulated Ideal Growth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
