from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lemmatization import lemmatizeData
from vectorRepresentation import createTFIDFVectorRepresentations, createBagOfWordsVectorRepresentations, createWord2VecVectorRepresentations
from modelTraining import trainModel, evaluateModel
import pandas as pd
from itertools import product
from sklearn.decomposition import TruncatedSVD


def main(lemmatize=False, createVector="", maxNumberOfFiles=None, drawPlots=False):
    inputDir = 'data'
    outputDir = 'dataProcessed'
    if lemmatize:
        lemmatizeData(inputDir, outputDir)

    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = f'{outputDir}/train'
    testFolderPath = f'{outputDir}/test'
    trainMatrixFile = f"trainingMatrix{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    testMatrixFile = f"testMatrix{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    if createVector == "TFIDF":
        createTFIDFVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "BagOfWords":
        createBagOfWordsVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "Word2Vec":
        createWord2VecVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)

    # # Wczytaj dane
    trainData = pd.read_csv(trainMatrixFile, header=None).values
    testData = pd.read_csv(testMatrixFile, header=None).values

    # Redukcja wymiarowości za pomocą PCA
    if createVector != "Word2Vec":
        print("🔄 Redukcja wymiarowości danych za pomocą PCA...")
        pca = TruncatedSVD(n_components=2500)  # Zredukuj do 2500 wymiarów
        trainDataReduced = pca.fit_transform(trainData)
        testDataReduced = pca.transform(testData)
        print(f"✅ Zredukowano wymiarowość do {pca.n_components} komponentów.")
    else:
        print("⚠️ Pomijanie redukcji wymiarowości dla Word2Vec...")
        trainDataReduced = trainData
        testDataReduced = testData

    # Lista modeli z parametrami
    models = [
        (LinearSVC, "LinearSVC", [
            {"C": [0.15, 0.2, 0.25, 0.5, 1.0, 2.0]},
            {"max_iter": [1000, 100, 10000]},
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
            {"max_iter": [1000, 100, 10000]},
            {"tol": [1e-3, 1e-4]},
        ])
    ]

    results = []

    for trainModelFunction, modelName, paramDictList in models:
        if modelName in ["LinearSVC", "Logistic Regression"] and createVector != "Word2Vec":
            currentTrainData = trainDataReduced
            currentTestData = testDataReduced
        else:
            currentTrainData = trainData
            currentTestData = testData
        # Tworzenie wszystkich kombinacji parametrów
        allParams = {}
        for paramDict in paramDictList:
            allParams.update(paramDict)
        paramNames = list(allParams.keys())
        paramValues = list(allParams.values())

        for paramCombination in product(*paramValues):
            params = dict(zip(paramNames, paramCombination))
            print(f"\n{'=' * 10} Training and Evaluating {createVector} {modelName} with params {params} {'=' * 10}")
            movieClassifier = trainModel(trainFolderPath, currentTrainData, trainModelFunction, **params, maxNumberOfFiles=maxNumberOfFiles)
            print(f"\nEvaluation on Training Set ({createVector} - {modelName}):")
            trainAccuracy, trainAUC = evaluateModel(movieClassifier, trainFolderPath, currentTrainData, f"training {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            print(f"\nEvaluation on Test Set ({createVector} - {modelName}):")
            testAccuracy, testAUC = evaluateModel(movieClassifier, testFolderPath, currentTestData, f"test {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
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
    resultsDf.to_excel(f"resultsSummary{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.xlsx", index=True)

    # Wyświetlanie najlepszych wyników dla każdego modelu
    print(f"\n=== Best Results for Each Model for {createVector} ===")
    for modelName in resultsDf['Model'].unique():
        bestResult = resultsDf[resultsDf['Model'] == modelName].sort_values(by='Test Accuracy', ascending=False).iloc[0]
        print(f"\nBest result for {modelName}:")
        print(bestResult)


if __name__ == "__main__":
    main(maxNumberOfFiles=5000, createVector="TFIDF")
    main(maxNumberOfFiles=5000, createVector="BagOfWords")
    main(maxNumberOfFiles=5000, createVector="Word2Vec")

    # main(lemmatize=True, createVector="TFIDF", maxNumberOfFiles=None)
    # main()
