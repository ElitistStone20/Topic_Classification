import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
import os
from test_classifiers import train_classifiers
from overfitting_test import test_over_fitting
from developed_classifier import train_final_classifier


# Funciton to Vectorize the corpus using the TfidfVectorizer
def vectorize_corpus(data):
    print("Vectorizing Corpus")
    # Vectorize the corpus
    transformer = TfidfVectorizer(smooth_idf=False)
    tfidf = transformer.fit_transform(data)
    return tfidf


# Partitions the dataset into data and labels
def seperate_dataset(dataset):
    print("Partitioning dataset into data and labels")
    data = []
    labels = []
    for item in dataset:
        data.append(item[0])
        labels.append(item[1])
    return data, labels


# Function to preprocess and normalise the sentences in the dataset This includes the following:
# Converting every character to lowercase, removing stopwords and lemmatizing each word in the sentences
def preprocessing(dataset):
    print("Preprocessing data...")
    sw_nltk = stopwords.words('english')
    new_dataset = []
    lemmatizer = WordNetLemmatizer()
    for i in range(0, len(dataset)):
        sentence = dataset[i][0]
        sentence = sentence.lower()
        words = [lemmatizer.lemmatize(word) for word in sentence.split() if word not in sw_nltk]
        sentence = " ".join(words)
        new_dataset.append([sentence, dataset[i][1]])
    print("Preprocessing complete")
    return new_dataset


# Function to read every text file in each folder, assign a label to them and place them in a multi dimentional array
def read_folders():
    print("Reading folders...")
    directories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    dataset = []
    for directory in directories:
        for filename in os.scandir("bbc/" + directory):
            if filename.is_file():
                file = open(filename, "r")
                dataset.append([file.read(), directory])
    return dataset


def download_nltk_requirements():
    print("Downloading required nltk packages...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("Download finished! Please restart the script")


# Partitions the dataset into training and test sets
def split_dataset(data, labels):
    print("Shuffling and splitting data into train and test sets....")
    data, labels = shuffle(data, labels, random_state=0)
    split = round(len(labels) * 0.7)
    X_train, X_test = data[:split, :], data[split:, :]
    Y_train, Y_test = labels[:split], labels[split:]
    return X_train, X_test, Y_train, Y_test


# Function to evaluate the user's input and execute the desired scripts
def evaluate_user_input(data, labels):
    route = input("Please enter one of the follow:\n"
                  "Download NLTK requirements - 0\n"
                  "Train Multiple Classifiers and display results - 1\n"
                  "Test for over-fitting - 2\n"
                  "Train final classification model - 3\n"
                  "Exit - -1\n")
    if route == '1':
        X_train, X_test, Y_train, Y_test = split_dataset(data, labels)
        train_classifiers(X_train, Y_train, X_test, Y_test)
    elif route == '2':
        test_over_fitting(data, labels)
    elif route == '3':
        train_final_classifier(data, labels)
    elif route == '0':
        download_nltk_requirements()
        return
    elif route == '-1':
        print("Exiting...")
        return
    else:
        print("Invalid entry!")
        evaluate_user_input(data, labels)


# Main entry point for application
def main():
    dataset = read_folders()
    dataset = preprocessing(dataset)
    data, labels = seperate_dataset(dataset)
    data = vectorize_corpus(data)
    evaluate_user_input(data, labels)


if __name__ == '__main__':
    main()
