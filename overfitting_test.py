import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt


# Removes redundant features based on the computation of impurity-based features
def feature_selection(data, labels):
    # Define Linear SVC with default parameters
    clf = LinearSVC()
    # Train the SVC with the data set and label set provided
    clf = clf.fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    # Transform the dataset to remove redundant feature computed by the classifier
    return model.transform(data)


# Function to train the RBF SVM using KFold cross validation
def test_over_fitting(data, labels):
    print("Testing for over-fitting using KFold cross validation")
    # Define dictionary of scoring measurements
    scoring = {'accuracy_macro': make_scorer(accuracy_score),
               'precision_macro': make_scorer(precision_score, average='macro'),
               'recall_macro': make_scorer(recall_score, average='macro'),
               'f1_macro': make_scorer(f1_score, average='macro')}
    # Select features from the dataset
    data = feature_selection(data, labels)
    # Define the KFold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # Train the RBF SVM using KFold cross validation
    scores = cross_validate(SVC(gamma=2, C=1), data, labels, scoring=scoring, cv=cv, n_jobs=-1)

    plt.rcdefaults()
    # Define figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    X = np.arange(10)
    # Plot bars for each subplot, where each subplot corresponds to a performance measurement
    ax1.barh(X, scores['test_accuracy_macro'], color='blue', align='center')
    ax2.barh(X, scores['test_precision_macro'], color='orange', align='center')
    ax3.barh(X, scores['test_recall_macro'], color='green', align='center')
    ax4.barh(X, scores['test_f1_macro'], color='red', align='center')

    # Set titles of each subplot
    ax1.set_title('Accuracy of each fold')
    ax2.set_title('Precision of each fold')
    ax3.set_title('Recall of each fold')
    ax4.set_title('F1-Score of each fold')

    # Adjust the white space between the subplots
    plt.subplots_adjust(wspace=0.5)
    #Display
    plt.show()
