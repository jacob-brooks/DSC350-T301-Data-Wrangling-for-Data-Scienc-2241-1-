# %% [markdown]
# Exercise 1
# 
# Import the NumPy library and create a NumPy array called my_array with the following values:

# %%
import numpy as np

my_array = np.array([1, 2, 3, 4, 5])

# %% [markdown]
# Exercise 2
# 
# Print the shape of the my_array array.

# %%
print(my_array.shape)

# %% [markdown]
# Exercise 3
# 
# Access the element at index 2 of the my_array array.

# %%
print(my_array[2])

# %% [markdown]
# Exercise 4
# 
# Create a new NumPy array called my_2d_array with the following values:

# %%
my_2d_array = np.array([[1, 2, 3], [4, 5, 6]])

# %% [markdown]
# Exercise 5
# 
# Print the shape of the my_2d_array array.

# %%
print(my_2d_array.shape)

# %% [markdown]
# Exercise 6
# 
# Access the element at row 1, column 2 of the my_2d_array array.

# %%
print(my_2d_array[1, 2])

# %% [markdown]
# Exercise 7
# 
# Create a new NumPy array called my_string_array with the following values:

# %%
my_string_array = np.array(['Hello', 'World!'])

# %% [markdown]
# Exercise 8
# 
# Print the type of the my_string_array array.

# %%
print(type(my_string_array))

# %% [markdown]
# Exercise 9
# 
# Try to access an element of the my_string_array array using an integer index. What happens?

# %%
# This will raise an error:
# TypeError: 'numpy.ndarray' object does not support indexing with type 'int'
my_string_array[0]

This is because NumPy arrays of strings cannot be indexed using integer indices. Instead, you can use string slices to access elements of a NumPy array of strings. For example, to access the first character of the first string in the my_string_array array, you would use the following code:

# %%
my_string_array[0][0]


