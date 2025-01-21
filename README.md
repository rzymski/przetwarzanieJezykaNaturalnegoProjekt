___
**Klasyfikator recenzji stworzony w python 3.11**
___

## Zawartość REDME
1. :scroll: [Funkcjonalność](#funkcjonalność)
2. :cd: [Instrukcja instalacji](#instrukcja-instalacji)
3. :hammer_and_wrench: [Instrukcja trenowania/testowania modeli](#instrukcja-trenowaniatestowania-modeli)
4. :computer: [Instukcja głównej aplikacji](#instukcja-głównej-aplikacji)

# Funkcjonalność
**Klasyfikator semantyki recenzji**
1) [x] Lemantyzacja plików tekstowych
2) [x] Tworzenie wektorów TF-IDF, Bag of words, Word2Vec, Doc2Vec, Fast text 
3) [x] Redukcja wymiarów wektorów
4) [x] Trenowanie modeli
5) [x] Ocena modeli
6) [x] Zapisywanie wyników modeli do plików xlsx
7) [x] Zapisywanie wektorów, modeli i redukcji wymiarów do plików pkl
8) [x] Klasyfikator wykorzystujący konkretny model do oceny konkretnego tekstu
9) [x] Aplikacja z GUI wykorzystująca klasyfikator

# Instrukcja instalacji
Instalujemy wszystkie potrzebne biblioteki z pliku `requirements.txt`
```sh
pip install -r .\requirements.txt
```
Możliwe wystąpienie konfilktów wersji dla niektórych bibliotek wymagających sprzecznych wersji numpy, wtedy trzeba instalować ręcznie biblioteki np.
```sh
pip install gensim
```

# Instrukcja trenowania/testowania modeli
Skrypt **`mainForTraining.py`** służy do testowania skuteczności modeli i wytrenowania najlepszych modeli.

Przykładowe użycie do wytrenowania najlepszych modeli:
```sh
'''Section that creates the best chosen models'''
tfidfBestModels = [
	(LogisticRegression, "Logistic Regression", [{"C": [2.0]}, {"solver": ["lbfgs"]}, {"max_iter": [1000]}, {"tol": [1e-3]}]),
	(StackingClassifier, "Stacking Classifier", [
		{"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5))]]},
		{"final_estimator": [LogisticRegression()]},
		{"passthrough": [True]}
	]),
	(VotingClassifier, "Voting Classifier", [
		{"estimators": [[('lr', LogisticRegression(C=1.5, solver='saga', max_iter=100, tol=0.001)), ('rf', RandomForestClassifier(n_estimators=1500, max_depth=None, min_samples_split=5)), ('svc', SVC(C=1.0, probability=True))]]},
		{"voting": ["soft"]}
	]),
]
main(createVector=True, vectorName="TFIDF", models=tfidfBestModels, save=True)
bagOfWordsBestModels = [
	(LogisticRegression, "Logistic Regression", [{"C": [0.5]}, {"solver": ["saga"]}, {"max_iter": [100]}, {"tol": [1e-3]}]),
	(RandomForestClassifier, "Random Forest", [{"n_estimators": [1500]}, {"max_depth": [None]}, {"min_samples_split": [10]}])
]
main(createVector=True, vectorName="BagOfWords", models=bagOfWordsBestModels, save=True)
world2VecBestModels = [(LogisticRegression, "Logistic Regression", [{"C": [0.5]}, {"solver": ["lbfgs"]}, {"max_iter": [100]}, {"tol": [1e-3]}])]
main(createVector=True, vectorName="Word2Vec", models=world2VecBestModels, save=True)
```

Przykładowe użycie do sprawdzenia skuteczności różnych modeli w połączeniu z różnymi metodami wektoryzacji:
```sh
'''Testing Section using 5000 files'''
main(maxNumberOfFiles=5000, vectorName="TFIDF", createVector=True)
main(maxNumberOfFiles=5000, vectorName="BagOfWords", createVector=True)
main(maxNumberOfFiles=5000, vectorName="Word2Vec", createVector=True)
main(maxNumberOfFiles=5000, vectorName="Doc2Vec", createVector=True)
main(maxNumberOfFiles=5000, vectorName="FastText", createVector=True)
```

### Parametry funkcji main w **`mainForTraining.py`**
|     parametr     | opis                                                                                         |
|:----------------:|----------------------------------------------------------------------------------------------|
|    lemmatize     | decyduje czy chcemy zlematyzować pliki z data do dataProcessed                               |
|    vectorName    | nazwa wektora np. TFIDF, BagOfWords lub Word2Vec                                             |
|   createVector   | czy ma tworzyć nowy wektor, (potrzebne jeśli wektor jeszcze nie istnieje)                    |
| maxNumberOfFiles | maksymalna liczba plików na których ma uczyć i testować model                                |
|    drawPlots     | czy ma rysować wykresy przy ewaluacji modeli                                                 |
|      models      | modele, które ma trenować i testować, w przypadku nie podania będą ustawione domyślne modele |
|       save       | czy ma zapisywać do plików pkl modele, wektory i pca                                         |
|   outputFolder   | ścieżka do folderu w którym mają być pliki wyjściowe                                         |

# Instukcja głównej aplikacji
Skrypt **`mainApplication.py`** zawiera główną aplikacje z gui, która wykorzystuje **`classifier.py`** do klasyfikacji poszczególnych recenzji filmowych.<br>
Aplikacja wymaga już gotowych modeli i wektorów. Ścieżki do nich powinny się znajdować w pliku **`classifier-config.ini`**

#### Przykładowa struktura **`classifier-config.ini`** 
przykłady modeli (każdy model potrzebuje ścieżki do modelu, i opcjonalnie wektora oraz redukcji wektorów)
```ini
[TFIDF Logistic Regression]
model = models/TFIDF_LogisticRegression.pkl
vector = vectors/TFIDF.pkl
reducer = reducers/TFIDF_Reducer.pkl

[TFIDF Voting Classifier]
model = models/TFIDF_VotingClassifier.pkl
vector = vectors/TFIDF.pkl

[TFIDF Stacking Classifier]
model = models/TFIDF_StackingClassifier.pkl
vector = vectors/TFIDF.pkl

[Bag Of Words Logistic Regression]
model = models/BagOfWords_LogisticRegression.pkl
vector = vectors/BagOfWords.pkl
reducer = reducers/BagOfWords_Reducer.pkl

[Bag Of Words Random Forest]
model = models/BagOfWords_RandomForest.pkl
vector = vectors/BagOfWords.pkl

[Word2Vec Logistic Regression]
model = models/Word2Vec_LogisticRegression.pkl
vector = vectors/Word2Vec.pkl

[BERT LinearSVC]
model = models/BERT_LinearSVC.pkl
reducer = reducers/BERT_Reducer.pkl

[BERT Logistic Regression]
model = models/BERT_LogisticRegression.pkl
reducer = reducers/BERT_Reducer.pkl

[BERT StackingClassifier]
model = models/BERT_StackingClassifier.pkl
reducer = reducers/BERT_Reducer.pkl

[BERT VotingClassifier]
model = models/BERT_VotingClassifier.pkl
reducer = reducers/BERT_Reducer.pkl
```
