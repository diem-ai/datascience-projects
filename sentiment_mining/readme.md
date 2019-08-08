




## 1. Motivation

## 2. Data Exploration
### Data Souce
### Data Structure
### Data Distribution
### Preprocessing Data
## 3. Machine Learning Modeling
### Term Frequency Document Matrix
- Machine doesn't understand the text. We have to transform reviews into sparse matrix or term-document matrix.
- The term-document matrix then is a two-dimensional matrix whose rows are the terms and columns are the documents, so each entry (i, j) represents the frequency of term i in document j.
- For each entry in the matrix, the term frequency measures the number of times that term i appears in document j, and the inverse document frequency measures the number of documents in the corpus which contain term i. The tf-idf score is the product of these two metrics (tf*idf). So an entry's tf-idf score increases when term i appears frequently in document j, but decreases as the term appears in other documents.
### Logistic Regession
- It is one of popular classifier. We will fit the model with x_train, y_train and compute the training score. Then, we use the model to predict x_test and compute the test score. Based on score of train & test, we can evaluate the accuracy and the effectiveness (overfitting) of the model.
### Cross Validation
- Cross Validation is a very useful technique for assessing the effectiveness of our model, particularly in cases where we need to mitigate overfitting. It is applied to more subsets created using the training dataset and each of which is used to train and evaluate a separate model
- In this project, I shall split x_train into 3 folds. We have 3 models runing through 3 subsets and the ith model will be built on the union of all subsets except the ith
### Randome Forest with Default Parameters
- A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the - predictive accuracy and control over-fitting.
- We imported scikit-learn RandomForestClassifier method to model the training dataset with random forest classifier.
### Tuning parameter with GridSearch CV
#### Conclusion
## 4. Topic Modeling with Latent Dirichlet Allocation




