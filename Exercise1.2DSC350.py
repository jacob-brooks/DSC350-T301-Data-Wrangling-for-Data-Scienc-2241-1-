# %%
"""
Title: Python Refresher Tasks
Author: Jacob Brooks
Date: 24 September 2023
Modified By: Jacob Brooks
Description: This program demonstrates basic Python tasks such as arithmetic operations, string manipulation, list, and tuple creation.
"""
# Display the text
print("Hello World! I wonder why that is always the default coding text to start with")

# Add two numbers together
num1 = 10
num2 = 5
result_addition = num1 + num2
print("Addition result:", result_addition)

# Subtract a number from another number
result_subtraction = num1 - num2
print("Subtraction result:", result_subtraction)

# Multiply two numbers
result_multiplication = num1 * num2
print("Multiplication result:", result_multiplication)

# Divide between two numbers
result_division = num1 / num2
print("Division result:", result_division)

# Concatenate two strings
string1 = "Hello, "
string2 = "world!"
concatenated_string = string1 + string2
print("Concatenated string:", concatenated_string)

# Create a list of 4 items
my_list = [1, "apple", 3.14, "banana"]
print("List:", my_list)

# Append an item to the list
my_list.append("cherry")
print("Updated list:", my_list)

# Create a tuple with 4 items
my_tuple = (5, "orange", 2.718, "grape")
print("Tuple:", my_tuple)

# %%
'''
Exercise 1
Create a Pandas DataFrame from the following list of data:
'''

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

import pandas as pd

# Create a DataFrame
df = pd.DataFrame(data, columns=["a", "b", "c"])

# Print the DataFrame
print(df)

# %%
'''
Exercise 2
Select the first two columns of the DataFrame.
'''
# Select the first two columns
df_subset = df[["a", "b"]]

# Print the subset DataFrame
print(df_subset)

# %%
'''
Exercise 3
Select the rows where the a column is greater than 5.
'''
# Select the rows where a > 5
df_filtered = df[df["a"] > 5]

# Print the filtered DataFrame
print(df_filtered)

# %%
'''
Exercise 4
Sort the DataFrame by the b column in descending order.
'''
# Sort the DataFrame by b in descending order
df_sorted = df.sort_values(by="b", ascending=False)

# Print the sorted DataFrame
print(df_sorted)

# %%
'''
Exercise 5
Calculate the sum of the values in the c column.
'''
# Calculate the sum of the values in the c column
sum_c = df["c"].sum()

# Print the sum
print(sum_c)

# %%
'''
Exercise 6
Group the DataFrame by the a column and calculate the mean of the values in the b column for each group.
'''
# Group the DataFrame by the a column and calculate the mean of the values in the b column for each group
df_grouped = df.groupby("a")["b"].mean()

# Print the grouped DataFrame
print(df_grouped)

# %%
'''
Exercise 7
Create a new column in the DataFrame called d and assign it the value 1 for all rows.
'''
# Create a new column in the DataFrame called d and assign it the value 1 for all rows
df["d"] = 1

# Print the DataFrame
print(df)

# %%
'''
Exercise 8
Drop the d column from the DataFrame.
'''
# Drop the d column from the DataFrame
df.drop("d", axis=1, inplace=True)

# Print the DataFrame
print(df)


