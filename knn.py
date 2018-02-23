
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

# Import the dataset from sklearn.
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Convert to Pandas dataframe for some exploration.
df = pd.DataFrame(columns=data.feature_names, data=data.data)
df['outcome'] = data.target

# Split data into test and training data with a test size of 30% (.3)
train_features, test_features, train_outcome, test_outcome = train_test_split(
    data.data, data.target, test_size=.3, random_state=11)

# Normalize and scale data (only features data, not outcome).
# Because KNN uses a distance based metric to compute similarity,
# it's important to normalize each column to the same scale before running the algorithm.
scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.fit_transform(test_features)

# Picking the best number of neighbors (K)

# Crete a validation set by sampling 20% (.2) out of the "training" data.
train_features_small, validation_features, train_outcome_small, validation_outcome = train_test_split(
    train_features, train_outcome, test_size=0.2, random_state=11)

# Assess accuracies of K from 1 through 10. To do this, loop through values of K,
# and in each loop:
# - Create a new classifier using K as the number of neighbors,
# - Fit the classifier to the (small) training data (without validation data).
# - Generate a set of predictions using the validation data.
# - Compute the accuracy of your model on your validation data.
best_k = 0
best_accuracy = 0

for k in np.arange(1, 10):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_fit = knn_clf.fit(train_features_small, train_outcome_small)
    knn_preds = knn_fit.predict(validation_features)
    accuracy = accuracy_score(knn_preds, validation_outcome)
    if (accuracy > best_accuracy).any():
        best_accuracy = accuracy
        best_k = k

print('BEST K:', best_k)

# Create a KNN classifier that uses the best nearest points.
knn_clf = KNeighborsClassifier(n_neighbors=best_k)

# K-Fold Cross Validation
# How accurately a predictive model will perform in practice.
# The shortcoming of this validation technique is that it only runs a single model
# across K different folds of data.
# If our case, the single model is when parameter k = 6.
# But what happens to model with parameter k = 1, 2, 3...?
cv = np.mean(cross_val_score(knn_clf, train_features,
                             train_outcome, cv=KFold(n_splits=10, shuffle=True, random_state=11)))
print('K-FOLD CV:', cv)

# Grid Search Cross Validation
# This model validation technique addresses the issue of simple cross validation.

# Build model with unscaled data.
param_grid = {'n_neighbors': np.arange(1, 50)}
# Generate two grid search so that we can compare difference between models generated from scaled and unscaled training features data.
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid)

# Fit the grid search to the training data.
grid_search.fit(train_features, train_outcome)
grid_search.best_estimator_

# Assess the actual accuracy of the model with test data.
# (Test data should only be used at the end after our model is made, tested, validated)
accuracy = grid_search.score(test_features, test_outcome)
print('MODEL ACCURACY (ORIGINAL):', accuracy)

# Visualze performance across neighbors K values.
# Now we can see which model with what k value generates the most accurate score.
test_scores = grid_search.cv_results_['mean_test_score']
plt.plot(param_grid['n_neighbors'], test_scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Average Accuracy Across CV Folds')
plt.show()

# Build model with scaled data.
param_grid_scaled = {'n_neighbors': np.arange(1, 50)}
grid_search_scaled = GridSearchCV(KNeighborsClassifier(), param_grid)
grid_search_scaled.fit(train_features_scaled, train_outcome)
grid_search_scaled.best_estimator_

accuracy_scaled = grid_search_scaled.score(
    test_features_scaled, test_outcome)
print('MODEL ACCURACY (SCALED):', accuracy_scaled)

test_scores_scaled = grid_search_scaled.cv_results_['mean_test_score']
plt.plot(param_grid['n_neighbors'], test_scores_scaled)
plt.xlabel('Number of Neighbors')
plt.ylabel('Average Accuracy Across CV Folds')
plt.show()

# Build model with scaled data using pipe.
# Integrate the data scaling into our cross validation process with pipeline (recommended).
pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier())

# Pass your pipeline to a grid search, specifying a set of neighbors to assess.
param_grid_pipe = {'kneighborsclassifier__n_neighbors': np.arange(1, 50)}
grid_search_pipe = GridSearchCV(pipe, param_grid_pipe)
grid_search_pipe.fit(train_features, train_outcome)
grid_search_pipe.best_estimator_

accuracy_pipe = grid_search_pipe.score(test_features, test_outcome)
print('MODEL ACCURACY (PIPE)', accuracy_pipe)

test_scores_pipe = grid_search_pipe.cv_results_['mean_test_score']
plt.plot(param_grid['n_neighbors'], test_scores_pipe)
plt.xlabel('Number of Neighbors')
plt.ylabel('Average Accuracy Across CV Folds')
plt.show()
