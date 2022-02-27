from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

names = [
    "Linear SVC - KNN",
    "ExtraTrees - KNN",
    "Linear SVC - Linear SVM",
    "ExtraTree - Linear SVM",
    "Linear SVC - RBF SVM",
    "ExtraTree - RBF SVM",
    "Linear SVC - Gaussian Process",
    "ExtraTree - Gaussian Process",
    "Linear SVC - Categorical NB",
    "ExtraTree - Categorical NB",
    "Linear SVC - Ada Boost",
    "ExtraTree - Ada Boost",
    "Linear SVC - Random Forest",
    "ExtraTree - Random Forest",
    "Linear SVC - Decision Tree",
    "ExtraTree - Decision Tree",
    "Linear SVC - NN",
    "ExtraTree - NN"
]

pipeline = [
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', KNeighborsClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', KNeighborsClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', SVC(kernel="linear", C=0.025))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', SVC(kernel="linear", C=0.025))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', SVC(gamma=2, C=1))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', SVC(gamma=2, C=1))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', GaussianNB())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', GaussianNB())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', CategoricalNB())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', CategoricalNB())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', AdaBoostClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', AdaBoostClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', SVC(gamma=2, C=1))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', SVC(gamma=2, C=1))
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', RandomForestClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', RandomForestClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', DecisionTreeClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', DecisionTreeClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(LinearSVC())),
        ('classification', MLPClassifier())
    ]),
    Pipeline([
        ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
        ('classification', MLPClassifier())
    ])
]
