from lemmatization import lemmatizeData
from vectorRepresentation import createVectorRepresentations
from modelTraining import trainModelLinearSVC, trainModelRandomForest, trainModelNaiveBayes, trainModelLogisticRegression
from modelTraining import trainAndEvaluateModel

if __name__ == "__main__":
    # Ścieżki do folderów i plików wyjściowych
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile)

    # Lista modeli do przetestowania
    models = [
        (trainModelLinearSVC, "LinearSVC"),
        (trainModelRandomForest, "Random Forest"),
        (trainModelNaiveBayes, "Naive Bayes"),
        (trainModelLogisticRegression, "Logistic Regression")
    ]

    # Trening i ocena dla każdego modelu
    for trainFunction, modelName in models:
        trainAndEvaluateModel(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, trainFunction, modelName)
