# %%
import numpy as np
import pandas as pd
import json
from io import StringIO

# %%
# Exercise 1
np.random.seed(42)
arr1 = np.random.rand(3, 4)
np.savetxt("np.csv", arr1, delimiter=",")
# You can view the content of np.csv with the 'cat' command in the terminal

# %%
# Exercise 2
df1 = pd.read_csv("np.csv", header=None)
print("DataFrame from np.csv:")
print(df1)

# %%
# Write DataFrame to CSV
df1.to_csv("df1.csv", index=False)

# %%
# Exercise 3
np.random.seed(42)
arr2 = np.random.rand(365, 4)
np.savetxt("random_array.csv", arr2, delimiter=",")
print("Size of random_array.csv:", pd.read_csv("random_array.csv").size)

# Save array in NumPy format
np.save("random_array.npy", arr2)

# Load array
loaded_array = np.load("random_array.npy")
print("Shape of loaded array:", loaded_array.shape)
print("Size of loaded array file:", np.load("random_array.npy").nbytes, "bytes")

# Create DataFrame from array
df2 = pd.DataFrame(loaded_array)

# Write DataFrame to pickle
df2.to_pickle("df2.pkl")

# Read DataFrame from pickle
df_from_pickle = pd.read_pickle("df2.pkl")

# Print size of pickle
print("Size of df2.pkl:", df_from_pickle.memory_usage().sum(), "bytes")

# %%
# Exercise 4
json_string = '{"country":"Netherlands","dma_code":"0","timezone":"Europe/Amsterdam","area_code":"0","ip":"46.19.37.108","asn":"AS196752","continent_code":"EU","isp":"Tilaa V.O.F.","longitude":5.75,"latitude":52.5,"country_code":"NL","country_code3":"NLD"}'

# Parse JSON string
json_data = json.loads(json_string)

# Print values for the "country" column
print("Country:", json_data["country"])

# Overwrite the value for "Netherlands" with your choice
json_data["country"] = "USA"
print("Updated Country:", json_data["country"])

# %%
# Exercise 5
# Create a Pandas Series from JSON string
json_series = pd.read_json(json.dumps(json_data), typ='series')

# Change the country value again to your choice
json_series["country"] = "Canada"

# Convert Pandas Series to JSON string
new_json_string = json_series.to_json()

# %%
pip install beautifulsoup4

# %%
# Exercise 6
# The BeautifulSoup exercise involves web scraping and can't be executed here.

from bs4 import BeautifulSoup
import re

soup = BeautifulSoup(open('C:/Users/JacobBrooks/Downloads/loremIpsum.html'),"lxml")

print("First div\n", soup.div)
print("First div class", soup.div['class'])

print("First dfn text", soup.dl.dt.dfn.text)

for link in soup.find_all('a'):
   print("Link text", link.string, "URL", link.get('href'))

# Omitting find_all
for i, div in enumerate(soup('div')):
   print(i, div.contents)


#Div with id=official
official_div = soup.find_all("div", id="official")
print("Official Version", official_div[0].contents[2].strip())

print("# elements with class", len(soup.find_all(class_=True)))

tile_class = soup.find_all("div", class_="tile")
print("# Tile classes", len(tile_class))

print("# Divs with class containing tile", len(soup.find_all("div", class_=re.compile("tile"))))

print("Using CSS selector\n", soup.select('div.notile'))
print("Selecting ordered list list items\n", soup.select("ol > li")[:2])
print("Second list item in ordered list", soup.select("ol > li:nth-of-type(2)"))

print("Searching for text string", soup.find_all(text=re.compile("2014")))


