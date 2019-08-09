




## 1. Motivation
- Somehow, I'm interested in what customers think about products after they bought them. I want to know which products feedbacks are positive and negative in a month so that I can develop or adjust the marketing or promotion campaign. 
- I'm also interested in not only whether people are talking with a positive, neutral, or negative feedbacks about the product, but also which particular aspects or features of the product people talk about. 
## 2. Data Exploration
### Data Structure
### Data Distribution
### Preprocessing Data
- The reviews contains many unexpected characters such as puntuation, numeric or meaningless words. 
## 3. Machine Learning Pipeline
### Split Dataset into train/set
- Divide dataset into 80% train and 20% test. 
```
train, test = train_test_split(data, test_size=0.2, random_state=0)

x_train = train['clean_review']
y_train = train['sentiment']

x_test = test['clean_review']
y_test = test['sentiment']

```
### Term Frequency Document Matrix
- Machine doesn't understand the text. We have to transform reviews into sparse matrix or term-document matrix.
- The term-document matrix then is a two-dimensional matrix whose rows are the terms and columns are the documents, so each entry (i, j) represents the frequency of term i in document j.
- For each entry in the matrix, the term frequency measures the number of times that term i appears in document j, and the inverse document frequency measures the number of documents in the corpus which contain term i. The tf-idf score is the product of these two metrics (tf*idf). So an entry's tf-idf score increases when term i appears frequently in document j, but decreases as the term appears in other documents.
### Logistic Regession
- It is one of popular classifier. We will fit the model with x_train, y_train and compute the training score. Then, we use the model to predict x_test and compute the test score. Based on score of train & test, we can evaluate the accuracy and the effectiveness (overfitting) of the model.
### Cross Validation
- Cross Validation is a very useful technique for assessing the effectiveness of our model, particularly in cases where we need to mitigate overfitting. It is applied to more subsets created using the training dataset and each of which is used to train and evaluate a separate model
- In this project, I shall split x_train into 3 folds. We have 3 models runing through 3 subsets and the ith model will be built on the union of all subsets except the ith
### Tuning parameters with Gridsearch
- In Logistic Regression, there are many parameters and each parameter has several values. Using grid search to find the best parameters and their values that can minimize the error funtion. 

- Put all in pipeline with the execution in order:

```
def grid_search(x_train, y_train, pipeline,  parameters,n_cv):
    """
    Perform GridSearchCV
    
    Arg:
        train_x, train_y: train set samples
        parameters: classifier's parameters
        pipeline: Pipeline object with classifer
    
    Return classifier with best estimator
    
    """
    t0 = time()

    grid_search_tune = GridSearchCV(pipeline
                                    , parameters
                                    , cv=n_cv
                                    , n_jobs=-1)
    
    grid_search_tune.fit(x_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()    
    return grid_search_tune

pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(token_pattern='(\S+)'))
                        ,('clf', LogisticRegression())
                    ])
parameters = {
    "tfidf__max_df": (0.5, 0.75, 0.95),
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.001, 0.01, 0.1],
    "clf__solver":["newton-cg", "lbfgs", "liblinear"]
}

lrg_gridsearch = grid_search(x_train, y_train, pipeline, parameters, n_cv=3)
```

### Evaluation

### Randome Forest with Default Parameters
- A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the - predictive accuracy and control over-fitting.
- We imported scikit-learn RandomForestClassifier method to model the training dataset with random forest classifier.
#### Conclusion
## 4. Topic Modeling with Latent Dirichlet Allocation
## 5. Reference




