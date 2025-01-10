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
def main(lemmatize=False, createVector="", maxNumberOfFiles=None, drawPlots=False, models=None, save=False):
    inputDir = 'data'
    outputDir = 'dataProcessed'
    if lemmatize:
        lemmatizeData(inputDir, outputDir)

    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = f'{outputDir}/train'
    testFolderPath = f'{outputDir}/test'
    trainMatrixFile = f"trainingMatrix{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    testMatrixFile = f"testMatrix{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.csv"
    vectorizerFilePath = f"vectors/{createVector}{maxNumberOfFiles if maxNumberOfFiles else ''}.pkl"

    # Wybór metody wektoryzacji
    vectorizer = None
    if createVector == "TFIDF":
        vectorizer = createTFIDFVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "BagOfWords":
        vectorizer = createBagOfWordsVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "Word2Vec":
        vectorizer = createWord2VecVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "Doc2Vec":
        vectorizer = createDoc2VecVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)
    elif createVector == "FastText":
        vectorizer = createFastTextVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=maxNumberOfFiles)

    # Zapis wektoryzera do pliku
    if vectorizer and save:
        saveToPkl(vectorizer, vectorizerFilePath)

    # # Wczytaj dane
    trainData = pd.read_csv(trainMatrixFile, header=None).values
    print("📖 Wczytano dane treningowe")
    testData = pd.read_csv(testMatrixFile, header=None).values
    print("📖 Wczytano dane testowe")

    # Redukcja wymiarowości za pomocą PCA
    if createVector in ["TFIDF", "BagOfWords"]:
        print("🔄 Redukcja wymiarowości danych za pomocą PCA...")
        pca = TruncatedSVD(n_components=2500)  # Zredukuj do 2500 wymiarów
        trainDataReduced = pca.fit_transform(trainData)
        testDataReduced = pca.transform(testData)
        saveToPkl(pca, f"vectors/{createVector}_PCA.pkl")  # Zapisanie PCA
        print(f"✅ Zredukowano wymiarowość do {pca.n_components} komponentów.")
    else:
        print("⚠️ Pomijanie redukcji wymiarowości dla Word2Vec...")
        trainDataReduced = trainData
        testDataReduced = testData

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
        if modelName in ["LinearSVC", "Logistic Regression"] and createVector not in ["Word2Vec", "Doc2Vec", "FastText"]:
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

        for index, paramCombination in enumerate(product(*paramValues)):
            params = dict(zip(paramNames, paramCombination))
            print(f"\n{'=' * 10} Training and Evaluating {createVector} {modelName} with params {params} {'=' * 10}")
            movieClassifier = trainModel(trainFolderPath, currentTrainData, trainModelFunction, **params, maxNumberOfFiles=maxNumberOfFiles)
            if save:
                saveToPkl(movieClassifier, f"models/{createVector}_{modelName.replace(' ', '')}{maxNumberOfFiles if maxNumberOfFiles else ''}_{index}.pkl")
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
    # tfidfBestModels = [
    #     (LogisticRegression, "Logistic Regression", [{"C": [1.5, 2.0]}, {"solver": ["lbfgs"]}, {"max_iter": [1000]}, {"tol": [1e-3]}])
    # ]
    # main(createVector="TFIDF", models=tfidfBestModels, save=True)
    # bagOfWordsBestModels = [
    #     (LogisticRegression, "Logistic Regression", [{"C": [1.5]}, {"solver": ["saga"]}, {"max_iter": [100]}, {"tol": [1e-3]}]),
    #     (RandomForestClassifier, "Random Forest", [{"n_estimators": [1000]}, {"max_depth": [None]}, {"min_samples_split": [5]}])
    # ]
    # main(createVector="BagOfWords", models=bagOfWordsBestModels, save=True)
    # world2VecBestModels = [(LogisticRegression, "Logistic Regression", [{"C": [0.5]}, {"solver": ["liblinear"]}, {"max_iter": [1000]}, {"tol": [1e-3]}])]
    # main(createVector="Word2Vec", models=world2VecBestModels, save=True)

    main(maxNumberOfFiles=5000, createVector="TFIDF")
    main(maxNumberOfFiles=5000, createVector="BagOfWords")
    main(maxNumberOfFiles=5000, createVector="Word2Vec")
    main(maxNumberOfFiles=5000, createVector="Doc2Vec")
    main(maxNumberOfFiles=5000, createVector="FastText")

    # main(lemmatize=True, createVector="TFIDF", maxNumberOfFiles=None)
    # main()
