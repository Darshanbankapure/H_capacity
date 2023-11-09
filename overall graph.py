import matplotlib.pyplot as plt
import seaborn as sns
# Example data (replace these with your actual error metrics)
models = ['Linear Reg', 'Ridge&Lasso', 'LSTM', 'CNN', 'Gradient', 'KNN', 'Ada+KNN']
mse_values = [74726.03, 74726.25, 11505.86, 10207.586, 9312.98, 8866.28, 7592.055]
rmse_values = [273.36, 273.36, 107.27, 101.033, 96.50, 94.16, 86.13]
r_squared_values = [0.66, 0.66, 0.95, 0.953, 0.96, 0.96, 0.976]
mse_upper_limit = max(mse_values) * 1.2
rmse_upper_limit = max(rmse_values) * 1.2
# Create subplots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Bar graph for MSE
sns.barplot(ax=ax[0], x=models, y=mse_values)
ax[1].set_ylim(0, rmse_upper_limit)
ax[0].set_title('Mean Squared Error (MSE)')
ax[0].tick_params(labelrotation=45)

# Bar graph for RMSE
sns.barplot(ax=ax[1], x=models, y=rmse_values)
ax[1].set_ylim(0, rmse_upper_limit)
ax[1].set_title('Root Mean Squared Error (RMSE)')
ax[1].tick_params(labelrotation=45)

# Bar graph for R-squared
sns.barplot(ax=ax[2], x=models, y=r_squared_values)
ax[2].set_title('R-squared Values')
ax[2].tick_params(labelrotation=45)

# Show the plot
plt.tight_layout()
plt.show()
