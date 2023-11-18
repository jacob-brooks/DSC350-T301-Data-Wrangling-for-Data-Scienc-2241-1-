# %% [markdown]
# Exercise 1: Print the first 3 rows of the DataFrame

# %%
import pandas as pd

# Create a list of data
data = [
    ['California', 39237950, 163696],
    ['Texas', 29502935, 691735],
    ['Florida', 21993250, 139672],
    ['New York', 19542977, 54556],
    ['Pennsylvania', 12885258, 46055],
    ['Illinois', 12741003, 55543],
    ['Ohio', 11799448, 41230],
    ['Georgia', 10711904, 59429],
    ['North Carolina', 10403697, 53843],
    ['Michigan', 10077331, 90538],
]

# Create a DataFrame
df = pd.DataFrame(data, columns=['State', 'Population', 'Area'])

# Print the first 3 rows of the DataFrame
print(df.head(3))


# %% [markdown]
# Exercise 2: Print the last 3 rows of the DataFrame

# %%
# Print the last 3 rows of the DataFrame
print(df.tail(3))


# %% [markdown]
# Exercise 3: Print descriptive statistics of the DataFrame

# %%
print(df.describe())


# %% [markdown]
# Exercise 4: Calculate the mean of the 'Population' column

# %%
print(df['Population'].mean())


# %% [markdown]
# Exercise 6: Sort the DataFrame by 'Population' in descending order and print the top 3 rows

# %%
print(df.sort_values(by='Population', ascending=False).head(3))



