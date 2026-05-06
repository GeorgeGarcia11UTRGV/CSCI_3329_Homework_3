#Homework 3 Part 2
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
import numpy as np

models = {
    'Linear Classifier' : Perceptron(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN'               : KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB'        : GaussianNB(),
    'Neural Network'     : MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
}

rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)
results = {}

print(f"{'Algorithm':<20} | {'Mean Accuracy':<15} | {'Std Dev':<10}")
print("-" * 50)

for name, model in models.items():
    scores = cross_val_score(model, x_final, y_final, cv=rkf, scoring='accuracy', n_jobs=-1)
    
    mean_score = scores.mean()
    std_score = scores.std()
    results[name] = (mean_score, std_score)
    
    print(f"{name:<20} | {mean_score:.4f}          | {std_score:.4f}")