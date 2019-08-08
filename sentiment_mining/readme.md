




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
### Logistic Regession with Default Parameters
### Randome Forest with Default Parameters
### Tuning parameter with GridSearch CV
#### Conclusion
## 4. Topic Modeling with Latent Dirichlet Allocation




