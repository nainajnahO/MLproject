import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import file
df = pd.read_csv("train_dataset.csv")

# Describe
cover_type = df[' Forest Cover Type Classes']
elevation = df['Elevation (meters)']


print(cover_type)
print(elevation)