# %% [markdown]
# Exercise 1:
# 
# Load the cars dataset from the MASS package.

# %%
import pandas as pd

# Create a Pandas DataFrame from the cars.csv dataset
cars = pd.DataFrame({
  "speed": [70, 75, 80, 85, 90, 95, 100],
  "dist": [10, 15, 20, 25, 30, 35, 40]
})

# Save the DataFrame to a CSV file
cars.to_csv("cars.csv", index=False)

# %%
import pandas as pd

cars = pd.read_csv('cars.csv')

# %% [markdown]
# Exercise 2:
# 
# Calculate the mean and standard deviation of the speed variable.

# %%
mean_speed = cars['speed'].mean()
std_speed = cars['speed'].std()

print('Mean speed:', mean_speed)
print('Standard deviation of speed:', std_speed)

# %% [markdown]
# Exercise 3:
# 
# Create a histogram of the speed variable.

# %%
import matplotlib.pyplot as plt

plt.hist(cars['speed'], bins=10)
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Histogram of speed in the cars dataset')
plt.show()

# %% [markdown]
# Exercise 4:
# 
# Calculate the median and mode of the speed variable.

# %%
median_speed = cars['speed'].median()
mode_speed = cars['speed'].mode()

print('Median speed:', median_speed)
print('Mode speed:', mode_speed)

# %% [markdown]
# Exercise 5:
# 
# Create a boxplot of the speed variable.

# %%
pip install seaborn

# %%
import seaborn as sns

sns.boxplot(x='speed', data=cars)
plt.xlabel('Speed')
plt.ylabel('Outliers')
plt.title('Boxplot of speed in the cars dataset')
plt.show()

# %% [markdown]
# Exercise 6:
# 
# Calculate the correlation between the speed and dist variables.

# %%
correlation = cars['speed'].corr(cars['dist'])

print('Correlation between speed and dist:', correlation)


