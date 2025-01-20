import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from itertools import product
from sklearn.decomposition import PCA
from modelTraining import trainModel, evaluateModel
from utils import measureExecutionTime, saveToPkl


# Initialize BERT tokenizer and model globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


@measureExecutionTime
def generateBERTEmbeddingsFromFiles(folderWithFiles, maxNumberOfFiles=None):
    embeddings = []
    labels = []

    print(f"📖 Reading files from {folderWithFiles}...")
    for label, subfolder in enumerate(["neg", "pos"]):  # 0 = Negative, 1 = Positive
        subfolderPath = os.path.join(folderWithFiles, subfolder)
        for i, file in enumerate(os.listdir(subfolderPath)):
            if maxNumberOfFiles and i >= maxNumberOfFiles:
                break
            with open(os.path.join(subfolderPath, file), 'r', encoding='utf-8') as f:
                text = f.read()
            # Generate BERT embedding
            tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**tokens)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            embeddings.append(cls_embedding)
            labels.append(label)

    print(f"✅ Generated embeddings for {len(embeddings)} files.")
    return np.array(embeddings), np.array(labels)


@measureExecutionTime
def mainForBERT(trainFolderPath, testFolderPath, models=None, maxNumberOfFiles=None, save=False, reduceDimensionality=True, reducedNumberOfDimensions=256, outputFolder="output", drawPlots=False):
    os.makedirs(outputFolder, exist_ok=True)
    # Generate BERT embeddings for training and test sets
    print("🚀 Generating BERT embeddings for training...")
    trainData, trainLabels = generateBERTEmbeddingsFromFiles(trainFolderPath, maxNumberOfFiles)
    print("🚀 Generating BERT embeddings for testing...")
    testData, testLabels = generateBERTEmbeddingsFromFiles(testFolderPath, maxNumberOfFiles)

    # Optional: Reduce dimensionality using PCA or TruncatedSVD
    if reduceDimensionality:
        print(f"🔄 Reducing dimensionality to {reducedNumberOfDimensions} components...")
        reducer = PCA(n_components=reducedNumberOfDimensions)
        trainData = reducer.fit_transform(trainData)
        testData = reducer.transform(testData)
        if save:
            saveToPkl(reducer, f"{outputFolder}/BERT_DimensionalityReducer.pkl")  # Save reducer to .pkl
            print(f"💾 Dimensionality reducer saved to {outputFolder}/BERT_DimensionalityReducer.pkl")

    # Define default models if none provided
    if not models:
        models = [
            (VotingClassifier, "Voting Classifier", [
                {"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5)), ('svc', SVC(C=1.0, probability=True))],
                                [('lr', LogisticRegression(C=2.0, solver='lbfgs', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5)), ('svc', SVC(C=1.0, probability=True))]]},
                {"voting": ["soft"]}
            ]),
            (StackingClassifier, "Stacking Classifier", [
                {"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5))]]},
                {"final_estimator": [LogisticRegression()]},
                {"passthrough": [True]}
            ]),
            (LinearSVC, "LinearSVC", [
                {"C": [0.15, 0.2, 0.25, 0.5, 1.0, 2.0]},
                {"max_iter": [100, 1000, 10000]},
                {"loss": ["hinge", "squared_hinge"]},
                {"tol": [1e-3, 1e-4]},
            ]),
            (RandomForestClassifier, "Random Forest", [
                {"n_estimators": [500, 1000, 1500]},
                {"max_depth": [5, 10, None]},
                {"min_samples_split": [2, 5, 10]}
            ]),
            (LogisticRegression, "Logistic Regression", [
                {"C": [0.5, 1.0, 1.2, 1.5, 2.0]},
                {"solver": ["liblinear", "lbfgs", "saga"]},
                {"max_iter": [100, 1000, 10000]},
                {"tol": [1e-3, 1e-4]},
            ])
        ]

    results = []

    for trainModelFunction, modelName, paramDictList in models:
        for paramCombination in product(*[list(p.values())[0] for p in paramDictList]):
            params = dict(zip([list(p.keys())[0] for p in paramDictList], paramCombination))
            print(f"\n{'=' * 10} Training {modelName} with params: {params} {'=' * 10}")
            # Train the model
            classifier = trainModel(trainFolderPath, trainData, trainModelFunction, **params, maxNumberOfFiles=maxNumberOfFiles)
            # Save the model
            if save:
                modelPath = f"{outputFolder}/{modelName.replace(' ', '_')}_{'_'.join(map(str, paramCombination))}.pkl"
                saveToPkl(classifier, modelPath)
                print(f"💾 Saved model to {modelPath}")
            # Evaluate the model
            print(f"\nEvaluation on Training Set ({modelName}):")
            trainAccuracy, trainAUC = evaluateModel(classifier, trainFolderPath, trainData, f"training {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            print(f"\nEvaluation on Test Set ({modelName}):")
            testAccuracy, testAUC = evaluateModel(classifier, testFolderPath, testData, f"test {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            # Append results
            results.append({
                "Model": modelName,
                "Parameters": params,
                "Train Accuracy": trainAccuracy,
                "Train AUC": trainAUC,
                "Test Accuracy": testAccuracy,
                "Test AUC": testAUC,
            })
    # Save and display results
    resultsDf = pd.DataFrame(results)
    resultsFile = f"{outputFolder}/BERT_results_summary.xlsx"
    resultsDf.to_excel(resultsFile, index=False)
    print(f"✅ Results saved to {resultsFile}")
    print("\n=== Best Results ===")
    print(resultsDf.sort_values(by="Test Accuracy", ascending=False).head())


if __name__ == "__main__":
    mainForBERT(trainFolderPath="data/train", testFolderPath="data/test", maxNumberOfFiles=5000, outputFolder="outputBert")
