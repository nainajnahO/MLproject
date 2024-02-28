import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from import_covertype_data import get_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import time


from sklearn.neighbors import KNeighborsClassifier


df = get_data("train_dataset.csv")

classes = df['Forest Cover Type Classes']

# absolue maximum scaling
df_abs_max_scaled = df.copy()
for column in df.columns:
    df_abs_max_scaled[column] = df[column] / df[column].abs().max()
    df_abs_max_scaled['Forest Cover Type Classes'] = classes

# min max scaling
scaler = MinMaxScaler()
df_min_max_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_min_max_scaled['Forest Cover Type Classes'] = classes

# normalization
df_normalized = df.copy()
for column in df.columns:
    df_normalized[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    df_normalized['Forest Cover Type Classes'] = classes
# Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_standardized['Forest Cover Type Classes'] = classes

# Robust Scaling
scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_robust_scaled['Forest Cover Type Classes'] = classes

scale_method_names = [
    'absolue maximum scaling',
    'min max scaling',
    'normalization',
    'Standardization',
    'Robust Scaling',
    'no_scaling'
]

scaled_data = [
    df_abs_max_scaled,
    df_min_max_scaled,
    df_normalized,
    df_standardized,
    df_robust_scaled,
    df
]

acc_dict = {}

for name, scaled_data in zip(scale_method_names, scaled_data):
    
    y = scaled_data['Forest Cover Type Classes']

    X = scaled_data.drop('Forest Cover Type Classes', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf = KNeighborsClassifier(3)
    
    start_time = time.time()
    
    clf.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    
    print(f'{name}, acc: {acc:.2f}, elapsed time:{elapsed_time:.2f} seconds')
    
    acc_dict[name] = acc
    

def get_scaled_data():
    '''
    return: list of scaled data, list of scaled data
    '''
    df = get_data("train_dataset.csv")

    # absolue maximum scaling
    df_abs_max_scaled = df.copy()
    for column in df.columns:
        df_abs_max_scaled[column] = df[column] / df[column].abs().max()


    # min max scaling
    scaler = MinMaxScaler()
    df_min_max_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # normalization
    df_normalized = df.copy()
    for column in df.columns:
        df_normalized[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    # Standardization
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


    # Robust Scaling
    scaler = RobustScaler()
    df_robust_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    scale_method_names = [
        'no_scaling'
        'absolue maximum scaling',
        'min max scaling',
        'normalization',
        'Standardization',
        'Robust Scaling'
    ]

    scaled_data = [
        df,
        df_abs_max_scaled,
        df_min_max_scaled,
        df_normalized,
        df_standardized,
        df_robust_scaled
    ]
    return scale_method_names, scaled_data

