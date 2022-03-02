import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Partitions the dataset into training and test sets
def split_dataset(data, labels):
    print("Shuffling and splitting data into train and test sets....")
    # Shuffles both the data and label set at the same time with
    # no randomness to ensure labels remain with their respective documents
    data, labels = shuffle(data, labels, random_state=0)
    # Define the split point as the index so that 70% of the data is allocated to the training set
    split = round(len(labels) * 0.7)
    # Partition the dataset using the split point
    X_train, X_test = data[:split, :], data[split:, :]
    Y_train, Y_test = labels[:split], labels[split:]
    return X_train, X_test, Y_train, Y_test


# Removes redundant features based on the computation of impurity-based features using a Linear SVC
def feature_selection(data, labels):
    # Define the Linear SVC with default parameters
    clf = LinearSVC()
    # Train the Linear SVC with the data provided and label set
    clf = clf.fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    # Transform the dataset removing redundant features
    return model.transform(data)


# Main function imported from other files.
# This function trains the final version of the RBF SVM
def train_final_classifier(data, labels):
    print("Training classifier...")
    # Selects relevant features from the dataset provided
    data = feature_selection(data, labels)
    # Partitions the dataset and label set 70/30
    X_train, X_test, Y_train, Y_test = split_dataset(data, labels)
    # Define the RBF SVM
    clf = SVC(gamma=2, C=1)
    # Train the SVM with the training set
    clf.fit(X_train, Y_train)
    # Predict the labels for the test set
    predictions = clf.predict(X_test)
    # Evaluate the performance of the SVM
    accuracy, recall, precision, f1 = accuracy_score(Y_test, predictions), \
                                      recall_score(Y_test, predictions, average='macro'), \
                                      precision_score(Y_test, predictions, average='macro'), \
                                      f1_score(Y_test, predictions, average='macro')
    print("Classifier trained")
    # Configure the figure size
    fig = plt.figure(figsize=(10, 5))
    # Plot the performance results on the figure as a bar graph
    plt.bar(['Accuracy', 'Recall', 'Precision', 'F1'], [accuracy, recall, precision, f1], color='red', width=0.6)
    plt.xlabel("Performance Measurement")
    plt.ylabel("Performance")
    plt.title("Performance of the final RBF SVM")
    # Display
    plt.show()
