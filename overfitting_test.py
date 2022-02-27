from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_validate


#
def feature_selection(data, labels):
    clf = LinearSVC()
    clf = clf.fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(data)


# Function to train the RBF SVM using KFold cross validation
def test_over_fitting(data, labels):
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    data = feature_selection(data, labels)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_validate(SVC(gamma=2, C=1), data, labels, scoring=scoring, cv=cv, n_jobs=-1)
    print(scores)
