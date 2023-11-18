# %% [markdown]
# # Title: Using the file WHO_first9cols.csv
# # Author: Jacob Brooks
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: This code will load the data from the CSV file into a DataFrame and then perform the requested tasks.

# %%
import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('C:/Users/JacobBrooks/Downloads/WHO_first9cols.csv')

# Task 1: Print the DataFrame
print("Task 1: Data in DataFrame")
print(df)

# Task 2: Query the number of rows
num_rows = len(df)
print("\nTask 2: Number of Rows:", num_rows)

# Task 3: Print the column headers
column_headers = df.columns.tolist()
print("\nTask 3: Column Headers:", column_headers)

# Task 4: Print the data types
data_types = df.dtypes
print("\nTask 4: Data Types:")
print(data_types)

# Task 5: Print the index
index = df.index
print("\nTask 5: Index:")
print(index)

# %% [markdown]
# # Title: Using the file WHO_first9cols.csv
# # Author: Jacob Brooks
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: This code will select the "Country" column, and then it will retrieve and print its data type, shape, index, values, and name.

# %%
import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('C:/Users/JacobBrooks/Downloads/WHO_first9cols.csv')

# Select the "Country" column
country_column = df['Country']

# Get the data type of the series
data_type = country_column.dtype

# Get the shape of the series
shape = country_column.shape

# Get the index of the series
index = country_column.index

# Get the values of the series
values = country_column.values

# Get the name of the series
name = country_column.name

# Print the results
print("Data Type:", data_type)
print("Series Shape:", shape)
print("Index:", index)
print("Values:", values)
print("Name:", name)

# %%
pip install quandl

# %%
import quandl

# Set your API key
quandl.ApiConfig.api_key = "znFVp4Tys_xGFUG5gy3T"

# Define the dataset you want to access
dataset_code = "SIDC/SUNSPOTS_A"

# Import the data
data = quandl.get(dataset_code)

# Print the head() and tail()
print("Head:")
print(data.head())
print("\nTail:")
print(data.tail())

# Query for the last value using the last date
last_value = data.iloc[-1]
print("\nLast Value:")
print(last_value)

# Query the date with a partial datetime string
date_string = '2023-01-01'  # Replace with the desired date in 'YYYY-MM-DD' format
date_data = data[data.index.strftime('%Y-%m-%d') == date_string]
print(f"\nData for {date_string}:")
print(date_data)

# Calculate the mean number of observations and sunspots
mean_observations = data['Number of Observations'].mean()
mean_sunspots = data['Yearly Mean Total Sunspot Number'].mean()

# Query with a Boolean, where the number of observations is greater than the mean number of observations
observations_greater_than_mean = data[data['Number of Observations'] > mean_observations]

# Query with a Boolean, where the number of sunspots is greater than the mean number of sunspots
sunspots_greater_than_mean = data[data['Yearly Mean Total Sunspot Number'] > mean_sunspots]

# Print the results
print("\nObservations Greater Than Mean:")
print(observations_greater_than_mean)

print("\nSunspots Greater Than Mean:")
print(sunspots_greater_than_mean)

# %%
import quandl
import numpy as np
import pandas as pd

# Set your Quandl API key
quandl.ApiConfig.api_key = "znFVp4Tys_xGFUG5gy3T"

# Define the dataset code
dataset_code = "SIDC/SUNSPOTS_A"

# Retrieve data from Quandl
data = quandl.get(dataset_code)

# Filter out rows where 'Sunspots' is not NaN
filtered_data = data.dropna(subset=['Yearly Mean Total Sunspot Number'])

# Calculate the requested statistics
describe_stats = filtered_data['Yearly Mean Total Sunspot Number'].describe()
count_observations = filtered_data['Yearly Mean Total Sunspot Number'].count()
mad_value = np.abs(filtered_data['Yearly Mean Total Sunspot Number'] - filtered_data['Yearly Mean Total Sunspot Number'].mean()).mean()
mean_value = filtered_data['Yearly Mean Total Sunspot Number'].mean()
median_value = filtered_data['Yearly Mean Total Sunspot Number'].median()
max_value = filtered_data['Yearly Mean Total Sunspot Number'].max()
min_value = filtered_data['Yearly Mean Total Sunspot Number'].min()
mode_value = filtered_data['Yearly Mean Total Sunspot Number'].mode().iloc[0]
std_deviation = filtered_data['Yearly Mean Total Sunspot Number'].std()
variance = filtered_data['Yearly Mean Total Sunspot Number'].var()
skewness = filtered_data['Yearly Mean Total Sunspot Number'].skew()

# Print the results
print("Descriptive Statistics:")
print(describe_stats)
print("\nCount of Observations:", count_observations)
print("\nMean Absolute Deviation (MAD):", mad_value)
print("\nMean:", mean_value)
print("\nMedian:", median_value)
print("\nMax:", max_value)
print("\nMin:", min_value)
print("\nMode:", mode_value)
print("\nStandard Deviation:", std_deviation)
print("\nVariance:", variance)
print("\nSkewness:", skewness)

# %%
import pandas as pd
import numpy as np

np.random.seed(10)
df = pd.DataFrame({'Weather': np.random.choice(['Hot', 'Cold'], 10),
                   'Food Price': np.random.randint(10, 50, 10),
                   'Number': np.random.randint(100, 200, 10)})

# Convert the 'Weather' column to a numeric type
df['Weather'] = df['Weather'].map({'Hot': 1, 'Cold': 0})

# Group the DataFrame by the 'Weather' column
grouped_df = df.groupby('Weather')

# Print the mean number and price for each group
print(grouped_df.agg(['mean']))

# %%
def g2(df):
    for name, group in df.groupby(['Weather', 'Food Price']):
        print(group.agg(['mean', 'median']))

g2(df.copy())

# %%
import pandas as pd
import numpy as np

np.random.seed(10)
df = pd.DataFrame({'Weather': np.random.choice(['Hot', 'Cold'], 10),
                   'Food Price': np.random.randint(10, 50, 10),
                   'Number': np.random.randint(100, 200, 10)})

# Select the first 3 rows
df1 = df.head(3)

# Put the first 3 rows back with the original DataFrame using the 'concat()' function
df2 = pd.concat([df1, df])

# Take those 3 rows and the last 2 rows of the original DataFrame and bring them together using the 'concat()' function
df3 = pd.concat([df1, df.tail(2)], ignore_index=True)

# Print the DataFrames
print(df1)
print(df2)
print(df3)

# %%
import pandas as pd

# Read the CSV files into DataFrames
dest_df = pd.read_csv('C:/Users/JacobBrooks/Downloads/dest.csv')
tips_df = pd.read_csv('C:/Users/JacobBrooks/Downloads/tips.csv')

# Merge DataFrames using the merge() function
merged_df = pd.merge(dest_df, tips_df, on='EmpNr')

# Print the merged results
print("Merged Data Using merge() function:")
print(merged_df)

# Alternatively, you can use the join() function
joined_df = dest_df.set_index('EmpNr').join(tips_df.set_index('EmpNr'))

# Print the joined results
print("\nMerged Data Using join() function:")
print(joined_df)

# %%
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('C:/Users/JacobBrooks/Downloads/WHO_first9cols.csv')

# Select the first 3 rows, including headers, for the specified columns
subset = df[['Country', 'Net primary school enrolment ratio male (%)']].head(3)

# Check for missing values
missing_values = subset.isnull()

# Count the number of NaN values
nan_count = missing_values.sum()

# Print non-missing values
non_missing_values = subset.dropna()
print("Non-missing values:")
print(non_missing_values)

# Replace missing values with a scalar (e.g., 0)
subset_filled = subset.fillna(0)

print("\nAfter replacing missing values with 0:")
print(subset_filled)


