import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# def loadLabelsFromFolders(folder, maxNumberOfFiles=None):
#     labels = []
#     for subFolder in ['neg', 'pos']:
#         folderPath = os.path.join(folder, subFolder)
#         for index, fileName in enumerate(os.listdir(folderPath), 1):
#             if fileName.endswith('.txt'):
#                 labels.append(0 if subFolder == 'neg' else 1)
#             if maxNumberOfFiles and index >= maxNumberOfFiles:
#                 print(f"✔️ Załadowano {maxNumberOfFiles=} labeli.")
#                 break
#     print(f"✔️ Wczytano wszystkie etykiety (łącznie: {len(labels)}).")
#     return labels

def loadLabelsFromFolders(folder, maxNumberOfFiles=None):
    labels = []
    for subFolder in ['neg', 'pos']:
        folderPath = os.path.join(folder, subFolder)
        numFiles = len([file for file in os.listdir(folderPath) if file.endswith('.txt')])
        if maxNumberOfFiles:
            numFiles = min(numFiles, maxNumberOfFiles)
            print(f"✔️ Ograniczono liczbę plików w {subFolder} do {maxNumberOfFiles}.")
        labels.extend([0 if subFolder == 'neg' else 1] * numFiles)
    print(f"✔️ Wczytano wszystkie etykiety (łącznie: {len(labels)}).")
    return labels


def trainModel(folderWithFiles, matrixData, modelTrainFunction, maxNumberOfFiles=None, **modelParams):
    # Load labels and data
    trainLabels = loadLabelsFromFolders(folderWithFiles, maxNumberOfFiles)
    trainTfidfDF = pd.read_csv(matrixData, header=None)
    # Initialize the model with the given parameters
    classifier = modelTrainFunction(**modelParams)
    # Train the model
    classifier.fit(trainTfidfDF, trainLabels)
    print(f"✅ Model {modelTrainFunction.__name__} trained successfully with parameters: {modelParams}")
    return classifier


def evaluateModel(classifier, folderWithFiles, dataMatrix, title="", maxNumberOfFiles=None):
    # Wczytanie etykiet i danych testowych
    labels = loadLabelsFromFolders(folderWithFiles, maxNumberOfFiles)
    tfidfDF = pd.read_csv(dataMatrix, header=None)
    # Przewidywanie i ocena wyników
    predictedLabels = classifier.predict(tfidfDF)
    accuracy = accuracy_score(labels, predictedLabels)
    print(f"Accuracy: {accuracy:.2f}")
    # Raport klasyfikacji
    report = classification_report(labels, predictedLabels, target_names=['Negative', 'Positive'])
    print("\nClassification Report:")
    print(report)
    # Macierz pomyłek (wykres)
    confMatrix = confusion_matrix(labels, predictedLabels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix of {title}")
    plt.show()


if __name__ == "__main__":
    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    # Lista modeli do przetestowania
    models = [
        (LinearSVC, "LinearSVC", {}),
        (RandomForestClassifier, "Random Forest", {"n_estimators": 100, "random_state": 42}),
        (MultinomialNB, "Naive Bayes", {}),
        (LogisticRegression, "Logistic Regression", {"max_iter": 1000, "random_state": 42})
    ]

    maxNFiles = 1000
    # Trening i ocena dla każdego modelu
    for trainModelFunction, modelName, trainModelParams in models:
        print(f"\n{'=' * 10} Training and Evaluating {modelName} {'=' * 10}")
        movieClassifier = trainModel(trainFolderPath, trainMatrixFile, trainModelFunction, **trainModelParams, maxNumberOfFiles=maxNFiles)
        print(f"\nEvaluation on Training Set ({modelName}):")
        evaluateModel(movieClassifier, trainFolderPath, trainMatrixFile, f"training {modelName}", maxNumberOfFiles=maxNFiles)
        print(f"\nEvaluation on Test Set ({modelName}):")
        evaluateModel(movieClassifier, testFolderPath, testMatrixFile, f"test {modelName}", maxNumberOfFiles=maxNFiles)
