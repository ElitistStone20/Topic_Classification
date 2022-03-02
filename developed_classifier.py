import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Partitions the dataset into training and test sets
def split_dataset(data, labels):
    print("Shuffling and splitting data into train and test sets....")
    data, labels = shuffle(data, labels, random_state=0)
    split = round(len(labels) * 0.7)
    X_train, X_test = data[:split, :], data[split:, :]
    Y_train, Y_test = labels[:split], labels[split:]
    return X_train, X_test, Y_train, Y_test


# Removes redundant features based on the computation of impurity-based features
def feature_selection(data, labels):
    clf = LinearSVC()
    clf = clf.fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(data)


def train_final_classifier(data, labels):
    print("Training classifier...")
    data = feature_selection(data, labels)
    X_train, X_test, Y_train, Y_test = split_dataset(data, labels)
    clf = SVC(gamma=2, C=1)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    accuracy, recall, precision, f1 = accuracy_score(Y_test, predictions), \
                                      recall_score(Y_test, predictions, average='macro'), \
                                      precision_score(Y_test, predictions, average='macro'), \
                                      f1_score(Y_test, predictions, average='macro')
    print("Classifier trained")
    fig = plt.figure(figsize=(10, 5))
    plt.bar(['Accuracy', 'Recall', 'Precision', 'F1'], [accuracy, recall, precision, f1], color='red', width=0.6)
    plt.xlabel("Performance Measurement")
    plt.ylabel("Performance")
    plt.title("Performance of the final RBF SVM")
    plt.show()
