

# Mining the sentiments in review's products & Unpacking the topics in customer's reviews

## 1. Motivation
- Somehow, I'm interested in what customers think about products after they bought them. I want to know which products feedbacks are positive and negative in a month so that I can develop or adjust the marketing or promotion campaign. 
- I'm also interested in not only whether people are talking with a positive, neutral, or negative feedbacks about the product, but also which particular aspects or features of the product people talk about. 
## 2. Data Exploration
### Data Structure
- Dataset has 183531 datapoints and 3 columns: name, review and rating
- Feature description:
    - name (string): product name
    - review (string): feeback of customers who bought the products
    - rating (integer): value is from 1 to 5 that shows how customers rate the product's quality
### Data Distribution
- Have a look at data structure, some questions come to my mind:
    - How are ratings distributed?  
    - Is there any difference of reviews length/words counts between ratings? I guess that people will leave posstive feebacks longer than negative feedbacks.
1) Ratings distribution

![](https://github.com/diem-ai/datascience-projects/blob/master/sentiment_mining/images/raings_distribution.PNG)

- Observation
From histogram chart, it is observed that most of the ratings are pretty high at 4 or 5 ranges. It means buyers thought that products are qualified

2) Review length distribution
- For each review, we will count how many words and add it as new feature for the data exploratory analysis and how length is distributed according to ratings.

```
products.dropna(inplace=True)
products['review_len'] = products['review'].apply(lambda x : len(x))

# Plot review length on histrogram with seaborn
sns.distplot(products['review_len'])
plt.title("Review Length Distribution")
plt.ylabel("Frequency")
plt.xlabel("Review Length")
plt.show()

products.groupby(['rating'])['review_len'].mean().plot(kind='barh')
plt.ylabel('Ratings')
plt.xlabel('Average of Review Length')
plt.show()

```
![](https://github.com/diem-ai/datascience-projects/blob/master/sentiment_mining/images/rev_len_dist.PNG)![](https://github.com/diem-ai/datascience-projects/blob/master/sentiment_mining/images/rev_len_rat_dist.PNG)

- Obseration
    - There were quite number of people like to leave long reviews
    - The higher rating was, the fewer words the reviews were
 
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
- The running process took around 10990.079s, we got the optimized parameters from Gris search:
```
lrg_gridsearch.best_estimator_

Pipeline(memory=None,
         steps=[('tfidf',
                 TfidfVectorizer(analyzer='word', binary=False,
                                 decode_error='strict',
                                 dtype=<class 'numpy.float64'>,
                                 encoding='utf-8', input='content',
                                 lowercase=True, max_df=0.5, max_features=None,
                                 min_df=1, ngram_range=(1, 1), norm='l2',
                                 preprocessor=None, smooth_idf=True,
                                 stop_words=None, strip_accents=None,
                                 sublinear_tf=False, token_pattern='(\\S+)',
                                 tokenizer=None, use_idf=True,
                                 vocabulary=None)),
                ('clf',
                 LogisticRegression(C=0.1, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1,
                                    l1_ratio=None, max_iter=100,
                                    multi_class='warn', n_jobs=None,
                                    penalty='l2', random_state=None,
                                    solver='liblinear', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False)
```


### Evaluation

### Randome Forest with Default Parameters
- A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the - predictive accuracy and control over-fitting.
- We imported scikit-learn RandomForestClassifier method to model the training dataset with random forest classifier.
#### Conclusion
## 4. Topic Modeling with Latent Dirichlet Allocation (LDA)
## 5. Reference
- [LDA explanation in Wiki](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
- Dataset and the lecture of Logistic Regression : Machine Learning Course - University of Washington
- [LDA's documentation by gensim](https://radimrehurek.com/gensim/models/ldamodel.html#usage-examples)



