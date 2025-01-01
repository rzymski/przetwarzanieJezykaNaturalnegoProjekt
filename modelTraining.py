import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from utils import measureExecutionTime


def loadLabelsFromFolders(folder, maxNumberOfFiles=None):
    labels = []
    for subFolder in ['neg', 'pos']:
        folderPath = os.path.join(folder, subFolder)
        numFiles = len([file for file in os.listdir(folderPath) if file.endswith('.txt')])
        if maxNumberOfFiles:
            numFiles = min(numFiles, maxNumberOfFiles)
            # print(f"✔️ Ograniczono liczbę plików w {subFolder} do {maxNumberOfFiles}.")
        labels.extend([0 if subFolder == 'neg' else 1] * numFiles)
    # print(f"✔️ Wczytano wszystkie etykiety (łącznie: {len(labels)}).")
    return labels


@measureExecutionTime
def trainModel(folderWithFiles, matrixData, modelTrainFunction, maxNumberOfFiles=None, **modelParams):
    # Load labels and data
    trainLabels = loadLabelsFromFolders(folderWithFiles, maxNumberOfFiles)
    if isinstance(matrixData, pd.DataFrame) or isinstance(matrixData, np.ndarray):
        train = matrixData
    else:
        train = pd.read_csv(matrixData, header=None).values
    # Initialize the model with the given parameters
    classifier = modelTrainFunction(**modelParams)
    # Train the model
    classifier.fit(train, trainLabels)
    print(f"✅ Model {modelTrainFunction.__name__} trained successfully with parameters: {modelParams}")
    return classifier


@measureExecutionTime
def evaluateModel(classifier, folderWithFiles, dataMatrix, title="", maxNumberOfFiles=None, drawPlots=False):
    # Wczytanie etykiet i danych testowych
    labels = loadLabelsFromFolders(folderWithFiles, maxNumberOfFiles)
    if isinstance(dataMatrix, pd.DataFrame) or isinstance(dataMatrix, np.ndarray):
        tfidfDF = dataMatrix
    else:
        tfidfDF = pd.read_csv(dataMatrix, header=None).values
    # Przewidywanie i ocena wyników
    predictedLabels = classifier.predict(tfidfDF)
    accuracy = accuracy_score(labels, predictedLabels)
    print(f"Accuracy: {accuracy:.2f}")
    # Obliczanie AUC-ROC
    if hasattr(classifier, "predict_proba"):  # Jeśli model obsługuje predict_proba
        predictedProba = classifier.predict_proba(tfidfDF)[:, 1]
        auc = roc_auc_score(labels, predictedProba)
    elif hasattr(classifier, "decision_function"):  # Jeśli model obsługuje decision_function (np. SVC)
        predictedProba = classifier.decision_function(tfidfDF)
        auc = roc_auc_score(labels, predictedProba)
    else:
        auc = None  # AUC-ROC nie może być obliczone dla tego modelu
    print(f"AUC-ROC: {auc:.2f}" if auc is not None else "AUC-ROC: Not available")
    # Raport klasyfikacji
    report = classification_report(labels, predictedLabels, target_names=['Negative', 'Positive'])
    # print("\nClassification Report:")
    # print(report)
    # Macierz pomyłek (wykres)
    if drawPlots:
        confMatrix = confusion_matrix(labels, predictedLabels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confMatrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix of {title}")
        plt.show()
    return accuracy, auc
