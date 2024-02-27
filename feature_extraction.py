import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


# models to compare
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


df = pd.read_csv('train_dataset.csv')


#-----------------------------------    acc comparison   -------------------------------------------------

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=0),
    SVC(gamma=2, C=1, random_state=0),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=0),
    DecisionTreeClassifier(max_depth=5, random_state=0),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=0
        ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=0),
    AdaBoostClassifier(algorithm="SAMME", random_state=0),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

rows_to_keep = int(len(df) * 0.01)

random_rows = df.sample(n=rows_to_keep, random_state=0)

df_remaining = df.drop(random_rows.index)

y = df_remaining[' Forest Cover Type Classes']

X = df_remaining.drop(' Forest Cover Type Classes', axis=1)

figure = plt.figure(figsize=(27, 9))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")
#------------------------------------------------------------------------------------

names = [] # the top preforming classifiers

classifiers = [] # the classifiers

importances = []

for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    importances.append(np.argsort(clf.feature_importances_)[::-1])
    
num_features = X_train.shape[1]
num_classifiers = len(classifiers)

fig, axs = plt.subplots(num_classifiers, 1, figsize=(10, 6*num_classifiers))

for i, (name, clf, importance) in enumerate(zip(names, classifiers, importances)):
    accuracies = []
    for num_removed in range(num_features // 2):
        # Remove features
        X_train_reduced = X_train.iloc[:, importance[:-num_removed]]
        X_test_reduced = X_test.iloc[:, importance[:-num_removed]]
        
        # Train and predict
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train_reduced, y_train)
        y_pred = clf.predict(X_test_reduced)
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
    # Plot accuracy vs. number of features removed
    axs[i].bar(range(num_features // 2), accuracies)
    axs[i].set_title(f"{name} - Accuracy vs. Features Removed")
    axs[i].set_xlabel("Number of Features Removed")
    axs[i].set_ylabel("Accuracy")
    
plt.tight_layout()
plt.show()