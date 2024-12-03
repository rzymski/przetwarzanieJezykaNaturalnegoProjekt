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


def loadLabelsFromFolders(folder, maxNumberOfFiles=1000):
    labels = []
    for subfolder in ['neg', 'pos']:
        folderPath = os.path.join(folder, subfolder)
        for index, fileName in enumerate(os.listdir(folderPath), 1):
            if fileName.endswith('.txt'):
                labels.append(0 if subfolder == 'neg' else 1)
            if index >= maxNumberOfFiles:
                print(f"✔️ Załadowano {maxNumberOfFiles=} labeli.")
                break
    print(f"✔️ Wczytano wszystkie etykiety (łącznie: {len(labels)}).")
    return labels


def trainModelLinearSVC(folderWithFiles, matrixData):
    trainLabels = loadLabelsFromFolders(folderWithFiles)
    trainTfidfDF = pd.read_csv(matrixData, header=None)
    classifier = LinearSVC()
    classifier.fit(trainTfidfDF, trainLabels)
    return classifier


def trainModelRandomForest(folderWithFiles, matrixData):
    trainLabels = loadLabelsFromFolders(folderWithFiles)
    trainTfidfDF = pd.read_csv(matrixData, header=None)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(trainTfidfDF, trainLabels)
    return classifier


def trainModelNaiveBayes(folderWithFiles, matrixData):
    trainLabels = loadLabelsFromFolders(folderWithFiles)
    trainTfidfDF = pd.read_csv(matrixData, header=None)
    classifier = MultinomialNB()
    classifier.fit(trainTfidfDF, trainLabels)
    return classifier


def trainModelLogisticRegression(folderWithFiles, matrixData):
    trainLabels = loadLabelsFromFolders(folderWithFiles)
    trainTfidfDF = pd.read_csv(matrixData, header=None)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(trainTfidfDF, trainLabels)
    return classifier


def evaluateModel(classifier, folderWithFiles, dataMatrix):
    # Wczytanie etykiet i danych testowych
    labels = loadLabelsFromFolders(folderWithFiles)
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
    plt.title("Confusion Matrix")
    plt.show()


def trainAndEvaluateModel(trainFolder, testFolder, trainMatrix, testMatrix, trainFunction, modelName):
    print(f"\n{'=' * 10} Training and Evaluating {modelName} {'=' * 10}")
    # Trening modelu
    classifier = trainFunction(trainFolder, trainMatrix)
    # Ocena zbioru treningowego
    print(f"\nEvaluation on Training Set ({modelName}):")
    evaluateModel(classifier, trainFolder, trainMatrix)
    # Ocena zbioru testowego
    print(f"\nEvaluation on Test Set ({modelName}):")
    evaluateModel(classifier, testFolder, testMatrix)


if __name__ == "__main__":
    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    # Lista modeli do przetestowania
    models = [
        (trainModelLinearSVC, "LinearSVC"),
        (trainModelRandomForest, "Random Forest"),
        (trainModelNaiveBayes, "Naive Bayes"),
        (trainModelLogisticRegression, "Logistic Regression")
    ]

    # Trening i ocena dla każdego modelu
    for trainModel, modelName in models:
        trainAndEvaluateModel(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, trainModel, modelName)
