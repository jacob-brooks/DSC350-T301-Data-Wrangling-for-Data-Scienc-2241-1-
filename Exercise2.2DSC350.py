# %% [markdown]
# # Title: vector addition the Python way
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: The first vector to be added contains the squares of 0 up to n using Python.

# %%
n = 3  # Replace with your desired value of n

# Using Python without NumPy
a = [i ** 2 for i in range(n + 1)]
b = [i ** 3 for i in range(n + 1)]

print("Vector a (squares):", a)
print("Vector b (cubes):", b)

# %% [markdown]
# # Title: vector addition the Python way
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: The first vector to be added contains the squares of 0 up to n using Numpy

# %%
import numpy as np

n = 3  # Replace with your desired value of n

# Using NumPy
a = np.arange(n + 1) ** 2
b = np.arange(n + 1) ** 3

print("Vector a (squares):", a)
print("Vector b (cubes):", b)

# %% [markdown]
# # Title: Iris Dataset Exploration
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: This program loads the Iris dataset, prints a description of the dataset, and plots sepal length vs. sepal width to    visualize the data.

# %%
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=iris.target, columns=['target'])

# Print a description of the dataset
print("Description of the Iris Dataset:")
print(data.describe())

# Plot sepal length vs. sepal width
plt.figure(figsize=(8, 6))
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=target['target'], cmap='viridis')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs. Sepal Width')
plt.colorbar(label='Target Class')
plt.show()

# %% [markdown]
# # Title: Boston Housing Dataset Exploration
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: This program loads the Boston Housing dataset, prints a description of the dataset, and plots the proportion of non-retail business vs. nitric oxide concentration to visualize the data.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the provided URL and format it
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Extract the columns of interest
x = data[:, 2]  # Proportion of non-retail business (Column 3)
y = data[:, 4]  # Nitric oxide concentration (Column 5)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, marker='+', color='b', label='Data Points')

# Label the axes
plt.xlabel("Proportion of Non-Retail Business (Column 3)")
plt.ylabel("Nitric Oxide Concentration (Column 5)")

# Add a title to the plot
plt.title("Scatter Plot of Boston Housing Dataset")

# Show a legend (optional)
plt.legend()

# Display the plot
plt.show()

# %% [markdown]
# # Title: creating a multidimensional array
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: This code creates a 2x3 multi-dimensional array, accesses and modifies elements within it. You can adjust the array dimensions and elements to suit your specific needs.

# %%
# Import the NumPy library
import numpy as np

# Create a 2x3 multi-dimensional array
# This array will have 2 rows and 3 columns
multi_dim_array = np.array([[1, 2, 3],
                            [4, 5, 6]])

# Print the multi-dimensional array
print("Multi-dimensional array:")
print(multi_dim_array)

# Accessing elements in the array
element = multi_dim_array[0, 1]  # Accessing the element at row 0, column 1
print("\nElement at (0, 1):", element)

# Changing an element in the array
multi_dim_array[1, 2] = 10  # Changing the element at row 1, column 2 to 10
print("\nUpdated multi-dimensional array:")
print(multi_dim_array)

# %% [markdown]
# # Title: an array of single precision floats
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: In this code, we explicitly specify the data type np.float32 when creating the NumPy array to ensure that it contains single-precision floating-point values. You can adjust the values in the array as needed.

# %%
# Import the NumPy library
import numpy as np

# Create an array of single-precision floats
single_precision_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# Print the array and its data type
print("Array of single-precision floats:")
print(single_precision_array)
print("Data type of the array:", single_precision_array.dtype)

# %% [markdown]
# # Title: an array of complex numbers
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: In this code, we use the np.complex64 data type to specify that the array should contain complex numbers represented as 64-bit floating-point values. You can adjust the complex numbers in the array as needed.

# %%
# Import the NumPy library
import numpy as np

# Create an array of complex numbers
complex_array = np.array([1 + 2j, 3 - 4j, 5 + 6j], dtype=np.complex64)

# Print the array and its data type
print("Array of complex numbers:")
print(complex_array)
print("Data type of the array:", complex_array.dtype)

# %% [markdown]
# # Title: Array work continued
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: In this code: We define the original array containing numbers from 0 to 8. We select a range of elements from indexes 3 to 7, which are elements 3 through 6 in the original array. We choose elements with an increment of 2 from index 0 to 7, resulting in [0, 2, 4, 6]. We reverse the chosen elements to get [6, 4, 2, 0].

# %%
# Import the NumPy library
import numpy as np

# Define an array from 0 to 8 (inclusive)
original_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Select elements from indexes 3 to 7 (3 through 6)
selected_range = original_array[3:7]

# Choose elements with an increment of 2 from index 0 to 7
chosen_elements = original_array[0:7:2]

# Reverse the chosen elements
reversed_array = chosen_elements[::-1]

# Print the results
print("Original Array:", original_array)
print("Selected Range (indexes 3 to 7):", selected_range)
print("Chosen Elements (index 0 to 7 with an increment of 2):", chosen_elements)
print("Reversed Array:", reversed_array)

# %% [markdown]
# # Title: create an array and perform the following functions
# # Author: Jacob Brooks
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: The ravel() and flatten() functions are equivalent. They both flatten the array into a 1D array. The reshape() function reshapes the array into the specified shape. However, it is important to note that the total number of elements in the array must remain the same. The T attribute transposes the array. This means that the rows and columns of the array are swapped. The resize() function changes the size of the array, but does not change the data. If the new size is smaller than the original size, the excess data is truncated. If the new size is larger than the original size, the array is padded with zeros.

# %%
import numpy as np

# Create an array
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Ravel
# Flattens the array into a 1D array
print("Ravel:", array.ravel())

# Flatten
# Same as Ravel
print("Flatten:", array.flatten())

# Setting the shape with a tuple
# Reshapes the array into the specified shape
print("Setting the shape with a tuple:", array.reshape((3, 3)))

# Transpose
# Swaps the rows and columns of the array
print("Transpose:", array.T)

# Resize
# Changes the size of the array, but does not change the data
print("Resize:", array.resize((2, 2)))

# %% [markdown]
# # Title: create an array and find the properties
# # Author: Jacob Brooks
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: The following Python code creates an array and finds the number of dimensions, count of elements, count of bytes, and full count of bytes

# %%
import numpy as np

# Create an array
array = np.array([1, 2, 3, 4, 5])

# Find the number of dimensions
ndims = array.ndim
print("Number of dimensions:", ndims)

# Find the count of elements
count_elements = array.size
print("Count of elements:", count_elements)

# Find the count of bytes
count_bytes = array.nbytes
print("Count of bytes:", count_bytes)

# Find the full count of bytes
full_count_bytes = array.itemsize * array.size
print("Full count of bytes:", full_count_bytes)

# %% [markdown]
# # Title:  convert an array to a Python list
# # Author: Armando Fandango
# # Date: 24 September 2023
# # Modified By: Jacob Brooks
# # Description: convert an array to a Python list, we can use the tolist() method. This method is provided by NumPy arrays, and it returns a list containing the same elements as the array.

# %%
import numpy as np

# Create a NumPy array
array = np.array([1, 2, 3, 4, 5])

# Convert the array to a list
list_from_array = array.tolist()

# Print the list
print(list_from_array)


