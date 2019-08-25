### Spam/ham classification
- Classify messages spam or ham
- Train supervised ML algorithms using document-term matrix (CountVectorizer, Tf-idf vectorizer) and Word2Vec

### Objectives
- Experience text classification with different approaches
- Compare the performance from supervised machine learning models using Bag-of-word (CountVectorizer) and Term frequency document maxtrix (Tf-idf Vectorizer)
- Compare the performance from supervised ML models using Word2Vec

### Project Notes
#### Dataset
- Got from  http://www.dt.fee.unicamp.br/~tiago/smsspamcollection
- Data Structure: 2 columns: label (target) and text (train data)
#### Code
###### <code> spam_classification.ipynb </code>: 
 * Explore dataset to anwser 5 questions:
 1) What percentage of the documents in spam_data are spam? 
 2) What is the longest token in the vocabulary in training set? 
 3) What is the average length of documents (number of characters) for not spam and spam documents? 
 4) What is the average number of digits per document for not spam and spam documents? 
 5) What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents? 
 * Create 3 new features for Feature Engineering;
 1) The average length of document in spam and ham documents
 2) The average number of digits per document for not spam and spam documents
 3) The number of bad characters in spam/ham documents
 * Fit and transform train data with <code>CountVectorizer</code> and <Tf-ifVectorizer> and compute the mean score from test data
 
###### <code> spam_classification_w2vec.ipynb </code>: 
- Clean the dataset and Lematization
- Train word2vec model from spam data
- Make vector of document from word2vec
- Fit classifiers with document vectors and compute the performance from predicting test data
- Plot score evaluation in bar chart

### Requirements
- Python >= 3.7
- Jupyter Notebook

### Dependencies
- pandas
- matplotlib
- scikit-learn
- numpy
- gensim
- Scipy
- nltk
- string
- WordCloud
- scipy
- warnings

### View the notebooks without installations
- Open <code>spam_classification_w2vec.html</code> and <code>spam_classification.htlm</code>

### Run notebooks on local
- Checkout the project : git clone https://github.com/diem-ai/spam_classification.git
- Install the latest version of libraries in requirements and dependencies

