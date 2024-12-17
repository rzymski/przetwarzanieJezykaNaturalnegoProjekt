from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from lemmatization import lemmatizeData
from vectorRepresentation import createVectorRepresentations
from modelTraining import trainModel, evaluateModel
import pandas as pd
from itertools import product


def main(lemmatize=False, createVectorTFIDF=False, maxNumberOfFiles=None, drawPlots=False):
    inputDir = 'data'
    outputDir = 'dataProcessed'
    if lemmatize:
        lemmatizeData(inputDir, outputDir)

    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = f'{outputDir}/train'
    testFolderPath = f'{outputDir}/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'
    if createVectorTFIDF:
        createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)

    # Lista modeli z parametrami
    models = [
        (LinearSVC, "LinearSVC", [
            {"C": [0.5, 1.0, 2.0]},
            {"max_iter": [1000, 100, 10000]},
            {"loss": ["hinge", "squared_hinge"]}
        ]),
        (RandomForestClassifier, "Random Forest", [
            {"n_estimators": [50, 100, 200]},
            {"max_depth": [5, 10, None]},
            {"min_samples_split": [2, 5, 10]}
        ]),
        (MultinomialNB, "Naive Bayes", [
            {"alpha": [0.1, 1.0, 2.0]},
            {"fit_prior": [True, False]},
            {"force_alpha": [True, False]}
        ]),
        (LogisticRegression, "Logistic Regression", [
            {"C": [0.5, 1.0, 2.0]},
            {"solver": ["liblinear", "lbfgs", "saga"]},
            {"max_iter": [1000, 100, 10000]}
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
            print(f"\n{'=' * 10} Training and Evaluating {modelName} with params {params} {'=' * 10}")
            movieClassifier = trainModel(trainFolderPath, trainMatrixFile, trainModelFunction, **params, maxNumberOfFiles=maxNumberOfFiles)
            print(f"\nEvaluation on Training Set ({modelName}):")
            trainAccuracy = evaluateModel(movieClassifier, trainFolderPath, trainMatrixFile, f"training {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            print(f"\nEvaluation on Test Set ({modelName}):")
            testAccuracy = evaluateModel(movieClassifier, testFolderPath, testMatrixFile, f"test {modelName} {params}", maxNumberOfFiles=maxNumberOfFiles, drawPlots=drawPlots)
            results.append({
                "Model": modelName,
                "Parameters": params,
                "Train Accuracy": trainAccuracy,
                "Test Accuracy": testAccuracy
            })

    # Wyświetlanie wyników w tabeli
    resultsDf = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("\n=== Summary of Results ===")
    print(resultsDf)
    resultsDf.to_csv("resultsSummary.csv", index=False)

    # Wyświetlanie najlepszych wyników dla każdego modelu
    print("\n=== Best Results for Each Model ===")
    for modelName in resultsDf['Model'].unique():
        best_result = resultsDf[resultsDf['Model'] == modelName].sort_values(by='Test Accuracy', ascending=False).iloc[0]
        print(f"\nBest result for {modelName}:")
        print(best_result)


if __name__ == "__main__":
    main(createVectorTFIDF=True, maxNumberOfFiles=1000)
    # main(lemmatize=True, createVectorTFIDF=True, maxNumberOfFiles=None)
    # main()
