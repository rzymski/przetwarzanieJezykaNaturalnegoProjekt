from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from lemmatization import lemmatizeData
from vectorRepresentation import createVectorRepresentations
from modelTraining import trainModel, evaluateModel


def main(lemmatize=False, createVectorTFIDF=False, maxNumberOfFiles=None):
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

    models = [
        (LinearSVC, "LinearSVC", {}),
        (RandomForestClassifier, "Random Forest", {"n_estimators": 100, "random_state": 42}),
        (MultinomialNB, "Naive Bayes", {}),
        (LogisticRegression, "Logistic Regression", {"max_iter": 1000, "random_state": 42})
    ]
    for trainModelFunction, modelName, trainModelParams in models:
        print(f"\n{'=' * 10} Training and Evaluating {modelName} {'=' * 10}")
        movieClassifier = trainModel(trainFolderPath, trainMatrixFile, trainModelFunction, **trainModelParams, maxNumberOfFiles=maxNumberOfFiles)
        print(f"\nEvaluation on Training Set ({modelName}):")
        evaluateModel(movieClassifier, trainFolderPath, trainMatrixFile, f"training {modelName}", maxNumberOfFiles=maxNumberOfFiles)
        print(f"\nEvaluation on Test Set ({modelName}):")
        evaluateModel(movieClassifier, testFolderPath, testMatrixFile, f"test {modelName}", maxNumberOfFiles=maxNumberOfFiles)


if __name__ == "__main__":
    main(lemmatize=True, createVectorTFIDF=True, maxNumberOfFiles=None)
    # main()
