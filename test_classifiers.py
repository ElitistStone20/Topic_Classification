from pipeline import pipeline, names
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np


# Function to train a variety of classification models
# Iterates over a pipeline of classifiers with different feature selection models
# Computes the accuracy, precision, recall and f1 score of each iteration
# Displays the results after computation
def train_classifiers(X_train, Y_train, X_test, Y_test):
    print("Training classifiers....")
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for name, clf in zip(names, pipeline):
        print("Training " + name + "....")
        clf.fit(X_train.toarray(), Y_train)
        predictions = clf.predict(X_test.toarray())
        accuracy.append(accuracy_score(Y_test, predictions))
        precision.append(precision_score(Y_test, predictions, average="macro", zero_division=1))
        recall.append(recall_score(Y_test, predictions, average="macro"))
        f1.append(f1_score(Y_test, predictions, average="macro"))
    print("Classification complete")
    print("Displaying results...")
    plt.rcdefaults()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    X = np.arange(len(names))
    ax1.barh(X, accuracy, color='blue', align='center')
    ax2.barh(X, precision, color='orange', align='center')
    ax3.barh(X, recall, color='green', align='center')
    ax4.barh(X, f1, color='red', align='center')

    ax1.set_yticks(X, labels=names)
    ax2.set_yticks(X, labels=names)
    ax3.set_yticks(X, labels=names)
    ax4.set_yticks(X, labels=names)

    ax1.set_title('Accuracy of trained classifiers')
    ax2.set_title('Precision of trained classifiers')
    ax3.set_title('Recall of trained classifiers')
    ax4.set_title('F1-Score of trained classifiers')

    plt.subplots_adjust(wspace=0.5)
    plt.show()

