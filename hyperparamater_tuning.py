from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from import_covertype_data import get_data
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the autoencoder model
def build_autoencoder(input_dim, encoding_dim):
    model = Sequential()
    model.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(input_dim, activation='relu'))
    return model

# Load dataset
df = get_data('train_dataset.csv')

X = df.drop('Forest Cover Type Classes', axis=1)
y = df['Forest Cover Type Classes']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define autoencoder parameters
input_dim = X_train.shape[1]
encoding_dim = 32  # Adjust according to the desired dimensionality of the encoded features

# Build and compile the autoencoder
autoencoder = build_autoencoder(input_dim, encoding_dim)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print('training the encoder...')
# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test), verbose=1)

# Extract features using the encoder part of the autoencoder
encoder = Sequential(autoencoder.layers[:1])
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)
print('encoder trained!')

print('scaling...')
scaler_encoded = MinMaxScaler()
X_train_encoded_scaled = scaler_encoded.fit_transform(X_train_encoded)
X_test_encoded_scaled = scaler_encoded.transform(X_test_encoded)
print('scaling done!')

# Define hyperparameter grids for each classifier
knn_param_grid = {'n_neighbors': range(1, 21),
                  'weights': ['uniform', 'distance']}

nn_param_grid = {'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
                 'activation': ['relu', 'tanh'],
                 'solver': ['adam'],
                 'alpha': [0.0001, 0.001, 0.01],
                 'learning_rate': ['constant', 'adaptive']}

dt_param_grid = {'max_depth': [None, 10, 20, 30, 40],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

rf_param_grid = {'n_estimators': [100, 200, 300],
                 'max_features': ['auto', 'sqrt'],
                 'max_depth': [None, 10, 20, 30],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4],
                 'bootstrap': [True, False]}

# Define classifiers
classifiers = [
    #('K-Nearest Neighbors', KNeighborsClassifier(), knn_param_grid),
    #('Neural Network', MLPClassifier(max_iter=1000), nn_param_grid),
    #('Decision Tree', DecisionTreeClassifier(), dt_param_grid),
    ('Random Forest', RandomForestClassifier(), rf_param_grid)
]

# Perform hyperparameter tuning for each classifier
for name, clf, param_grid in classifiers:
    print(f"Tuning hyperparameters for {name}...")
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=20, cv=5, verbose=2, n_jobs=-1)
    random_search.fit(X_train_encoded_scaled, y_train)
    
    print(f"Best parameters found for {name}:")
    print(random_search.best_params_)
    print(f"Best score: {random_search.best_score_}\n")