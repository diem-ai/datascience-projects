
# Mining sentiment review & Unpacking the topics in customer's reviews

## Table of content
1. [Motivation](/sentiment_mining#1-motivation)
2. [Project Description](/sentiment_mining#2-project-description)
3. [Data Exploration](/sentiment_mining#2-data-exploration)
    - [Data Structure](/sentiment_mining#data-structure)
    - [Data Distribution & Observation](/sentiment_mining#data-distribution)
    - [Processing Data](/sentiment_mining#processing-data)
3. [Machine Learning Pipeline](/sentiment_mining#3-machine-learning-pipeline)
    - [Train/Set Split](/sentiment_mining#split-dataset-into-trainset)
    - [Term Frequency Document Matrix](/sentiment_mining#term-frequency-document-matrix)
    - [Logistic Regession](/sentiment_mining#logistic-regession)
    - [Cross Validation](/sentiment_mining#cross-validation)
    - [Tuning parameters with Gridsearch](/sentiment_mining#tuning-parameters-with-gridsearch)
    - [Evaluation Metrics](/sentiment_mining#Evaluation)
4. [Topic Modeling with Latent Dirichlet Allocation (LDA)](/sentiment_mining#4-topic-modeling-with-latent-dirichlet-allocation-lda)
5. [Installation](/sentiment_mining#5-installation)
6. [Run code/Notebook in Local](/sentiment_mining#6-run-codenotebook-in-local)
7. [Run code/Notebook in Google Colab](/sentiment_mining#7-run-codenotebook-in-google-colab)
8. [Reference](/sentiment_mining#8-reference)

## 1. Motivation
- Somehow, I'm interested in what customers think about products after they bought them. I want to know which products feedbacks are positive and negative in a month so that I can develop or adjust the marketing or promotion campaign. 
- I'm also interested in not only whether people are talking with a positive, neutral, or negative feedbacks about the product, but also which particular aspects or features of the product people talk about. 
## 2. Project Description
- The goal is to explore logistic regression and feature engineering with existing sklearn functions and unpack the topics in customer's review. with Laten Dirichlet Allocation. I will use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative or neutral.
- Train a logistic regression model to predict the sentiment of product reviews.
- Inspect the weights (coefficients) of a trained logistic regression model.
- Make a prediction (both class and probability) of sentiment for a new product review.
- Evaluation model's performance with accuracy score and classification report
## 3. Data Exploration
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

![](/sentiment_mining/images/raings_distribution.PNG)

- Observation
    - From histogram chart, it is observed that most of the ratings are pretty high at 4 or 5 ranges. It means buyers thought that products are qualified

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
![](/sentiment_mining/images/rev_len_dist.PNG)![](/sentiment_mining/images/rev_len_rat_dist.PNG)

- Obseration
    - There were quite number of people like to leave long reviews
    - The higher rating was, the fewer words the reviews were
 
### Preprocessing Data
- Remove punctuation Python's built-in string functionality.
- Remove alpha numerical words
- Lower and remove stop words
- Lemmatizer: It reduces the inflected words properly ensuring that the root word belongs to the language. We need to provide the context in which you want to lemmatize that is the parts-of-speech (POS). This is done by giving the value for pos parameter in wordnet_lemmatizer.lemmatize

````
from text_accessory import processing_text
products['processed_review'] = products['review'].apply(processing_text)

````

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
- For each entry in the matrix, the term frequency measures the number of times that term i appears in document j, and the inverse document frequency measures the number of documents in the corpus which contain term i. The tf-idf score is the product of these two metrics (tf*idf). So an entry's tf-idf score increases when term i appears frequently in document j, but decreases as the term appears in other documents. In another word, ``idf`` is a cross-document normalization, that puts less weight on common terms, and more weight on rare terms
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
#### Classification Report
````
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

                precision    recall  f1-score   support

          -1       0.78      0.47      0.59      5259
           0       0.51      0.02      0.05      3328
           1       0.83      0.99      0.90     27954

    accuracy                           0.83     36541
   macro avg       0.71      0.49      0.51     36541
weighted avg       0.80      0.83      0.78     36541

````
- Observation:
    - Model predicts 83% accuracry on test set
    - Postive reviews (class = 1): 99% reviews are predicted as postive and 83% of them are predicted truly possitive
    - Negative reviews (class = -1): Model predicts 2% reviews negative and 51% of them are predicted truly negative
    - Netrual reviews (class = 0): 47% reviews are predicted neutral and model predicts 78% correctly.

#### Training score vs Test score

![](/sentiment_mining/images/train_vs_test.PNG)

- Observation: Logistic Regression Model performs a little bit better with test set and the gap betweem training score and testing score is not high. It can be said that Logistic Regression is not overfitting and it can learn data's pattern

#### Important features
````
def get_top_tfidf_words(feature_names, sample, top_n=2):  
  """
  feature_names: an array of words. for example: ["love", "great", "product"]
  sample: sparse matrix
  top_n: (int) number of selected words
  
  """
  # sort indices pf sample, backward the sorted array and select top_n items
  sorted_nzs = np.argsort(sample.data)[:-(top_n+1):-1]
  # return an sub-array with sorted indices
  return feature_names[sample.indices[sorted_nzs]]

feature_names = np.array(tfidf.get_feature_names())

top_n_words = get_top_tfidf_words(feature_names, tfidf.transform(x_test), 30)

print("Top 15 words : {}".format(top_n_words))

get_wordcloud(' '.join(top_n_words))

````
![](/sentiment_mining/images/top_n_words.PNG)

## 4. Topic Modeling with Latent Dirichlet Allocation (LDA)
- Topic modeling is a type of statistical model that is used to extract topics that are collections of words collection of documents. Latent Dirichlet Allocation is one of implementation of Topic Modelling.
- Latent Dirichlet allocation (LDA) is a topic model that generates topics based on word frequency from a set of documents. LDA is particularly useful for finding reasonably accurate mixtures of topics within a given document set.
- In this project, I will use `gensim.models.ldamodel` to cluster the similar topics in reviews.

````
#import library
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.corpora import Dictionary as gendict

# create dictionary(id, word) from document
dictionary = gendict([rev.split() for rev in reviews])
# create bag of word corpus from the dictionary and entire document
bow_corpus =  [dictionary.doc2bow(rev.split()) for rev in reviews]
# Train the model on the bag of word corpus
# 
lda = LdaModel(bow_corpus
               , num_topics=5 # numer of topics generated
              , id2word=dictionary
              , iterations = 100)
lda.print_topics()
````
- At the first try, the model will output 5 topics from the dataset as the result:
````
[(0,
  '0.016*"bag" + 0.011*"diaper" + 0.010*"like" + 0.009*"use" + 0.007*"great" + 0.006*"well" + 0.006*"diapers" + 0.006*"love" + 0.006*"case" + 0.005*"product"'),
 (1,
  '0.006*"product" + 0.006*"put" + 0.006*"baby" + 0.006*"back" + 0.005*"use" + 0.005*"like" + 0.005*"easy" + 0.004*"could" + 0.004*"open" + 0.004*"way"'),
 (2,
  '0.022*"baby" + 0.015*"great" + 0.014*"love" + 0.011*"little" + 0.010*"loves" + 0.010*"old" + 0.009*"like" + 0.008*"daughter" + 0.008*"cute" + 0.007*"well"'),
 (3,
  '0.030*"seat" + 0.018*"car" + 0.015*"stroller" + 0.013*"easy" + 0.008*"baby" + 0.008*"great" + 0.008*"love" + 0.008*"like" + 0.007*"back" + 0.007*"use"'),
 (4,
  '0.013*"monitor" + 0.012*"baby" + 0.010*"use" + 0.009*"bottle" + 0.009*"great" + 0.008*"bottles" + 0.007*"like" + 0.007*"cup" + 0.006*"pump" + 0.006*"time"')]
````
- Interpretation:
    - Topic 1: bags and diapers products and their quality is great
    - Topic 2: products are easy to use for baby
    - Topic 3: the same to topic 2
    - Topic 4: toys like: car, seat and stroller
    - Topic 5: another products of baby: monitor, bottles , cup and pumps. It is similar to topic 1

- Finally, we have 3 different topics at the first try. We will try with numer of topics = 10 to see if we can find out some other interesting topics.

````
lda = LdaModel(bow_corpus
               , num_topics=10
              , id2word=dictionary
              , iterations = 100)
 
topics = lda.print_topics(num_topics=10, num_words=10)

for (idx, topic) in topics:
  print("topic: {}\n{}".format(idx+1, topic))

topic: 1
0.026*"love" + 0.025*"great" + 0.022*"little" + 0.020*"cute" + 0.017*"baby" + 0.014*"perfect" + 0.013*"nice" + 0.012*"loves" + 0.012*"like" + 0.012*"well"
topic: 2
0.023*"stroller" + 0.009*"like" + 0.008*"easy" + 0.007*"back" + 0.006*"really" + 0.006*"use" + 0.005*"easily" + 0.005*"little" + 0.005*"side" + 0.005*"small"
topic: 3
0.033*"seat" + 0.019*"car" + 0.018*"old" + 0.012*"baby" + 0.010*"son" + 0.009*"loves" + 0.009*"daughter" + 0.008*"little" + 0.008*"like" + 0.007*"love"
topic: 4
0.031*"baby" + 0.020*"product" + 0.016*"quality" + 0.014*"great" + 0.013*"received" + 0.012*"gift" + 0.011*"good" + 0.011*"love" + 0.010*"made" + 0.010*"price"
topic: 5
0.027*"monitor" + 0.014*"baby" + 0.013*"night" + 0.009*"sound" + 0.008*"unit" + 0.008*"room" + 0.008*"time" + 0.008*"video" + 0.007*"product" + 0.007*"battery"
topic: 6
0.029*"bag" + 0.021*"diaper" + 0.013*"use" + 0.011*"diapers" + 0.010*"like" + 0.007*"bags" + 0.007*"cloth" + 0.006*"case" + 0.006*"ve" + 0.006*"dry"
topic: 7
0.021*"baby" + 0.020*"crib" + 0.017*"soft" + 0.013*"blanket" + 0.012*"sleep" + 0.011*"bed" + 0.011*"cover" + 0.010*"mattress" + 0.010*"pillow" + 0.009*"night"
topic: 8
0.008*"product" + 0.007*"britax" + 0.005*"you" + 0.005*"install" + 0.004*"back" + 0.004*"piece" + 0.004*"together" + 0.004*"box" + 0.004*"make" + 0.004*"put"
topic: 9
0.021*"bottle" + 0.018*"bottles" + 0.016*"cup" + 0.014*"pump" + 0.012*"cups" + 0.011*"use" + 0.009*"milk" + 0.009*"water" + 0.009*"baby" + 0.008*"like"
topic: 10
0.050*"easy" + 0.021*"food" + 0.020*"use" + 0.019*"clean" + 0.019*"great" + 0.019*"chair" + 0.018*"love" + 0.012*"tray" + 0.011*"high" + 0.010*"baby"
````
- Intepretation
    - topic 1: possitive feedbacks for baby products
    - topic 2: stroller and easy use
    - topic 3: baby girls love car toys
    - topic 4: products have good price. It is quite duplicated with topic 1
    - topic 5: video game
    - topic 6: baby cloths
    - topic 7: sleepping products
    - topic 8: seating products for baby (britax). It is similar to topic 2
    - topic 9: cater products (bottles, cup, milk, water)
    - topic 10: good feedbacks. The same to topic 1
- At the end, we have 8 topics after tunning parameters and the result sounds interesting.

## 5. Installation
- Software requirement:
    - Python >= 3.7
    - Jupyter Notebook
- Dependencies:
    - pandas
    - matplotlib
    - numpy
    - scikit-learn
    - wordcloud
    - nltk
    - gensim
## 6. Run code/Notebook in Local
In a terminal or command window, navigate to the top-level project directory sentiment_mining/ (that contains this README) and run one of the following commands:

````
ipython notebook sentiment_prediction2.ipynb
ipython notebook sentiment_LDA.ipynb.ipynb
````
or
````
jupyter notebook sentiment_prediction2.ipynb
jupyter notebook sentiment_LDA.ipynb.ipynb
````
## 7. Run code/Notebook in Google Colab

https://colab.research.google.com/drive/1SdzL7lEXEXVg9yUFGWr8MQmyD4zzl50U#scrollTo=y8DAaQv49w7n&uniqifier=2
https://colab.research.google.com/drive/1E2pC8Vgh_uIPSiO9iHqLj7bLiFfFCxH8

## 8. Reference
- [LDA explanation in Wiki](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
- Dataset and the lecture of Logistic Regression : Machine Learning Course - University of Washington
- [LDA's documentation by gensim](https://radimrehurek.com/gensim/models/ldamodel.html#usage-examples)



