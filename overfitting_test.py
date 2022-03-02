import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt


# Removes redundant features based on the computation of impurity-based features
def feature_selection(data, labels):
    clf = LinearSVC()
    clf = clf.fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(data)


# Function to train the RBF SVM using KFold cross validation
def test_over_fitting(data, labels):
    print("Testing for over-fitting using KFold cross validation")
    scoring = {'accuracy_macro': make_scorer(accuracy_score),
               'precision_macro': make_scorer(precision_score, average='macro'),
               'recall_macro': make_scorer(recall_score, average='macro'),
               'f1_macro': make_scorer(f1_score, average='macro')}

    data = feature_selection(data, labels)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(SVC(gamma=2, C=1), data, labels, scoring=scoring, cv=cv, n_jobs=-1)
    print("Accuracy average: " + str(sum(scores['test_accuracy_macro'])/10) + "\n"
          "Precision Average: " + str(sum(scores['test_precision_macro'])/10) + "\n"
          "Recall Average: " + str(sum(scores['test_recall_macro'])/10) + "\n"
          "F1 Average: " + str(sum(scores['test_f1_macro'])/10))
    plt.rcdefaults()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    X = np.arange(10)
    ax1.barh(X, scores['test_accuracy_macro'], color='blue', align='center')
    ax2.barh(X, scores['test_precision_macro'], color='orange', align='center')
    ax3.barh(X, scores['test_recall_macro'], color='green', align='center')
    ax4.barh(X, scores['test_f1_macro'], color='red', align='center')

    ax1.set_title('Accuracy of each fold')
    ax2.set_title('Precision of each fold')
    ax3.set_title('Recall of each fold')
    ax4.set_title('F1-Score of each fold')

    plt.subplots_adjust(wspace=0.5)
    plt.show()
