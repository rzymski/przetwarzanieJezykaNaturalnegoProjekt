import os
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lemmatization import lemmatizeData
from vectorRepresentation import createTFIDFVectorRepresentations, createBagOfWordsVectorRepresentations, createWord2VecVectorRepresentations, createDoc2VecVectorRepresentations, createFastTextVectorRepresentations
from modelTraining import trainModel, evaluateModel
import pandas as pd
from itertools import product
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier, StackingClassifier
from utils import measureExecutionTime, saveToPkl


@measureExecutionTime
def main(lemmatize=False, vectorName="", createVector=False, maxNumberOfFiles=None, drawPlots=False, models=None, save=False, outputFolder='output'):
    inputDir = 'data'
    outputDir = 'dataProcessed'
    if lemmatize:
        lemmatizeData(inputDir, outputDir)

    os.makedirs(outputFolder, exist_ok=True)

    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = f'{outputDir}/train'
    testFolderPath = f'{outputDir}/test'
    trainMatrixFile = f"{outputFolder}/trainingMatrix{vectorName}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    testMatrixFile = f"{outputFolder}/testMatrix{vectorName}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    vectorizerFilePath = f"{outputFolder}/{vectorName}{maxNumberOfFiles if maxNumberOfFiles else ''}.pkl"

    # Wybór metody wektoryzacji
    vectorizer = None
    if vectorName == "TFIDF" and createVector:
        vectorizer = createTFIDFVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif vectorName == "BagOfWords" and createVector:
        vectorizer = createBagOfWordsVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif vectorName == "Word2Vec" and createVector:
        vectorizer = createWord2VecVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif vectorName == "Doc2Vec" and createVector:
        vectorizer = createDoc2VecVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif vectorName == "FastText" and createVector:
        vectorizer = createFastTextVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)

    # Zapis wektoryzera do pliku
    if vectorizer and save:
        saveToPkl(vectorizer, vectorizerFilePath)

    # Wczytaj dane
    trainData = pd.read_csv(trainMatrixFile, header=None).values
    print("📖 Wczytano dane treningowe")
    testData = pd.read_csv(testMatrixFile, header=None).values
    print("📖 Wczytano dane testowe")

    # Redukcja wymiarowości za pomocą PCA
    if vectorName in ["TFIDF", "BagOfWords"]:
        print("🔄 Redukcja wymiarowości danych za pomocą PCA...")
        reducer = TruncatedSVD(n_components=2500)  # Zredukuj do 2500 wymiarów
        trainData = reducer.fit_transform(trainData)
        testData = reducer.transform(testData)
        if save:
            saveToPkl(reducer, f"{outputFolder}/{vectorName}_Reducer.pkl")  # Zapisanie PCA
        print(f"✅ Zredukowano wymiarowość do {reducer.n_components} komponentów.")
    else:
        print(f"⚠️ Pomijanie redukcji wymiarowości dla {vectorName}...")

    if not models:
        # Lista modeli z parametrami
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
        # Tworzenie wszystkich kombinacji parametrów
        allParams = {}
        for paramDict in paramDictList:
            allParams.update(paramDict)
        paramNames = list(allParams.keys())
        paramValues = list(allParams.values())

        for paramCombination in product(*paramValues):
            params = dict(zip(paramNames, paramCombination))
            print(f"\n{'=' * 10} Training and Evaluating {vectorName} {modelName} with params {params} {'=' * 10}")
            movieClassifier = trainModel(trainFolderPath, trainData, trainModelFunction, **params, maxNumberOfFiles=maxNumberOfFiles)
            if save:
                saveToPkl(movieClassifier, f"{outputFolder}/{vectorName}_{modelName.replace(' ', '')}{maxNumberOfFiles if maxNumberOfFiles else ''}.pkl")
            print(f"\nEvaluation on Training Set ({vectorName} - {modelName}):")
            trainAccuracy, trainAUC = evaluateModel(movieClassifier, trainFolderPath, trainData, f"training {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            print(f"\nEvaluation on Test Set ({vectorName} - {modelName}):")
            testAccuracy, testAUC = evaluateModel(movieClassifier, testFolderPath, testData, f"test {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            results.append({
                "Model": modelName,
                "Parameters": params,
                "Train Accuracy": trainAccuracy,
                "Train AUC": trainAUC,
                "Test Accuracy": testAccuracy,
                "Test AUC": testAUC
            })

    # Wyświetlanie wyników w tabeli
    resultsDf = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\n=== Summary of Results ===")
    print(resultsDf)
    resultsDf.to_excel(f"{outputFolder}/resultsSummary{vectorName}{maxNumberOfFiles if maxNumberOfFiles else ''}.xlsx", index=True)

    # Wyświetlanie najlepszych wyników dla każdego modelu
    print(f"\n=== Best Results for Each Model for {vectorName} ===")
    for modelName in resultsDf['Model'].unique():
        bestResult = resultsDf[resultsDf['Model'] == modelName].sort_values(by='Test Accuracy', ascending=False).iloc[0]
        print(f"\nBest result for {modelName}:")
        print(bestResult)


if __name__ == "__main__":
    # '''Section that creates the best chosen models'''
    # tfidfBestModels = [
    #     (LogisticRegression, "Logistic Regression", [{"C": [2.0]}, {"solver": ["lbfgs"]}, {"max_iter": [1000]}, {"tol": [1e-3]}]),
    #     (StackingClassifier, "Stacking Classifier", [
    #         {"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5))]]},
    #         {"final_estimator": [LogisticRegression()]},
    #         {"passthrough": [True]}
    #     ]),
    #     (VotingClassifier, "Voting Classifier", [
    #         {"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5)), ('svc', SVC(C=1.0, probability=True))]]},
    #         {"voting": ["soft"]}
    #     ]),
    # ]
    # main(createVector=True, vectorName="TFIDF", models=tfidfBestModels, save=True, drawPlots=True)
    # bagOfWordsBestModels = [
    #     (LogisticRegression, "Logistic Regression", [{"C": [0.5]}, {"solver": ["saga"]}, {"max_iter": [100]}, {"tol": [1e-3]}]),
    #     (RandomForestClassifier, "Random Forest", [{"n_estimators": [1500]}, {"max_depth": [None]}, {"min_samples_split": [10]}])
    # ]
    # main(createVector=True, vectorName="BagOfWords", models=bagOfWordsBestModels, save=True, drawPlots=True)
    # world2VecBestModels = [(LogisticRegression, "Logistic Regression", [{"C": [0.5]}, {"solver": ["lbfgs"]}, {"max_iter": [100]}, {"tol": [1e-3]}])]
    # main(createVector=True, vectorName="Word2Vec", models=world2VecBestModels, save=True, drawPlots=True)

    '''Testing Section using 5000 files'''
    main(maxNumberOfFiles=5000, vectorName="TFIDF", createVector=True)
    main(maxNumberOfFiles=5000, vectorName="BagOfWords", createVector=True)
    main(maxNumberOfFiles=5000, vectorName="Word2Vec", createVector=True)
    main(maxNumberOfFiles=5000, vectorName="Doc2Vec", createVector=True)
    main(maxNumberOfFiles=5000, vectorName="FastText", createVector=True)
