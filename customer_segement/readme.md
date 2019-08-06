# Customer Segmentation

## Table of content
1. [Motivation](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#1-motivation)
2. [Data Exploration](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#2-data-exploration)
    - [Data Source](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#data-source)
    - [Data Structure](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#data-structure)
    - [Data Distribution](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#data-distribution)
    - [Normalization & Outliers Detection](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#normalization--outliers-detection)
3. [Feature Engineering](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#3-feature-engineering)
4. [Clustering with Agglomerative Clustering](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#4-clustering-with-agglomerative-clustering-algorithm)
5. [Files Description](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#5-file-description)
6. [Installation](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#6-installation)
7. [Run in Local](https://github.com/diem-ai/mlprojects/tree/master/customer_segement#7-run-in-local)

##

### 1. Motivation
- As a Data Analyst, I want to analyze customers’ interaction throughout the customer journey so that I'm enable to learn something about customers and thereby improve marketing opportunities and purchase rates. 
- One goal of the project is to describe the variation in the different types of customers that a wholesale distributor interacts with
and to predict dymaically segmentation so that I can design campaigns, creatives and schedule for product release
![](https://github.com/diem-ai/user-segment/blob/master/results/product_distribution.PNG)

### 2. Data Exploration
#### Data Source
The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal. More information can be found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
#### Data Structure
- Fresh: annual spending (m.u.) on fresh products (Continuous);
- Milk: annual spending (m.u.) on milk products (Continuous);
- Grocery: annual spending (m.u.) on grocery products (Continuous);
- Frozen: annual spending (m.u.) on frozen products (Continuous);
- Detergents_Paper: annual spending (m.u.) on detergents and paper products (Continuous);
- Delicatessen: annual spending (m.u.) on and delicatessen products (Continuous);
- Channel: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)
- Region: {Lisnon - 1, Oporto - 2, or Other - 3} (Nominal)
#### Data Distribution
![](https://github.com/diem-ai/user-segment/blob/master/results/data_describe.PNG)
- Observation: the mean & standard deviation are high among features. We can probably guess that data is noisy and skewed.
- How are sales of product distributed in channels?

![](https://github.com/diem-ai/user-segment/blob/master/results/channel_distributed.PNG)

- Observation: As we can see, Fresh is largely distributed in Hotel/Restaurant/Cafe meanwhile Grocery is highly propotioned in distribution of Retail, following is Milk, Fresh and Detergents_Paper products. Both channels sold very few Delicatessen and Frozen products

#### Normalization & Outliers Detection
- One of the big challenge of this project is to deal with outliers and skewed distribution. Anomalies, or outliers, can be a serious issue when training machine learning algorithms or applying statistical techniques. They are often the result of errors in measurements or exceptional system conditions and therefore do not describe the common functioning of the underlying system. Indeed, the best practice is to implement an outlier removal phase before proceeding with further analysis.

![](https://github.com/diem-ai/user-segment/blob/master/results/data_distribution.PNG)

- Observation:
    - From the scatter matrix, it can be observed that that the pair (Grocery, Detergents_Paper) seems to have the strongest correlation. The pair (Grocery, Milk)  and (Detegerns_Paper, Milk) salso seem to exhibit some degree of correlation. This scatter matrix also confirms my initial suspicions that Fresh, Frozen and Delicatessen product category donr not have significant correlations to any of the remaining features. 
    - Additionally, this scater matrix also show us that the data for these features is highly skewed and not normaly distributed.

- Normalizing data:  In this project, I will use `Box-Cox` test implemented from `scipy.stats.boxcox`. The `Box Cox` transformation is used to stabilize the variance (eliminate heteroskedasticity) and also to (multi)normalize a distribution. We shall observe the transformed data again in scatter plot to see how well it is rescaled.

![](https://github.com/diem-ai/user-segment/blob/master/results/BoxCox.PNG)

- Outliers Detection & Deletion: 
    - There are many techniques to detect and optionally remove outliers: Numeric Outlier, Z-Score and DBSCAN.
        - Numeric Outlier: This is the simplest, nonparametric outlier detection method in a one dimensional feature space. Outliers are calculated by means of the IQR (InterQuartile Range) with interquartile multiplier value k=1.5.
        - Z-score is a parametric outlier detection method in a one or low dimensional feature space. This technique assumes a Gaussian distribution of the data. The outliers are the data points that are in the tails of the distribution and therefore far from the mean
        - DBSCAN: This technique is based on the DBSCAN clustering method. DBSCAN is a non-parametric, density based outlier detection method in a one or multi dimensional feature space. This is a non-parametric method for large datasets in a one or multi dimensional feature space
    - I use <code>Numeric Outlier</code> method because it is simple and it can work well with normalized data. The process looks like:
   ```
   # select relevant features in dataset
    data =  data[features]

    # Create a dictionary to keep datapoints of outliers of each features
    outliers_indices = {}

    for label in features:

      # 25th percentile
      q1 = np.percentile(data[label], 25)

      # 75th percentile
      q3 = np.percentile(data[label], 75) 

      # Outlier step
      IQR =  (q3 - q1)*1.5

      # Lower Limit
      lower = q1 - IQR

      # Upper Limit
      upper = q3 + IQR

      # Outlier criteria
      not_good_criteria = ~ ((data[label] >= lower) & (data[label] <= upper) )

      # Outlier datapoints selected
      outliers =  data[not_good_criteria]

      # Add outliers of the current feature to dictionary
      outliers_indices.update({label: outliers.index.tolist()})

      print("Feature: {} , Lower Limit: {}, Upper Limit: {}".format(label, lower, upper))

      display(outliers)
  
   ```

### 3. Feature Engineering
- Using `PCA (Principal Components Analysis)` in sklearn to extract the important features in the dataset. When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained.
![](https://github.com/diem-ai/user-segment/blob/master/results/PCA.PNG)
- The plot above clearly shows that most of the variance (<b>87.48%</b> of the variance to be precise) can be explained by the first principal component alone. The second principal component still bears some information (<b>6.79%</b>) while the third, fourth, fifth and sixth principal components can safely be dropped without losing to much information. Together, the first two principal components contain 94.27% of the information.
- Now we redo PCA on dataset with only 2 components. Prior to that, it makes sense to standardize the data, especially, if it was measured on different scales. 2 in 1, we have the result below:

![](https://github.com/diem-ai/user-segment/blob/master/results/Standadize_PCA.PNG)

- Observation:
  - There is high correlation in spending of clients who buy Frozen and Fresh products.
  - However, clients who buy Grocery products who will buy Detergents_Paper and Milk products.
  - Delicatessen seems unrelates to other products.

### 4. Clustering with Agglomerative Clustering algorithm
- Agglomerative Clustering algorithm groups similar objects into groups called clusters. It recursively merges the pair of clusters that minimally increases a given linkage distance. It starts with many small clusters and merge them together to create bigger clusters:

    ![](https://github.com/diem-ai/user-segment/blob/master/results/likage.PNG)
    
- What is the number of cluster to be considered as a good parameter to algorithm for this case? Using `sklearn.metrics.silhouette_score` to calculate the distance between features and clusters. We choose the value with the highest score:
```
for i in range(2, len(features) + 1):
  model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
  model.fit(transformed_data)
  labels = model.labels_
  score = silhouette_score(transformed_data, labels, metric='euclidean')
  print("Cluster: {}, score: {}".format(i, score))

Cluster: 2, score: 0.31464135811015653
Cluster: 3, score: 0.299218354075286
Cluster: 4, score: 0.29332856881138303
Cluster: 5, score: 0.31285975761587975
Cluster: 6, score: 0.2978440686133343

```
- Clearly, Model returns the highest score with cluster=2. Fitting model with data which is transformed and plotting clustered data:

![](https://github.com/diem-ai/user-segment/blob/master/results/clustering.PNG)

### 5. File Description
- customer.csv: training dataset
- customer-segmentation.ipynb: Python code of data visualization, data wrangling and machine learning modeling

### 6. Installation
- Software requirement: 
    - Python >= 2.7
    - Jupyter Notebook
- Dependencies:
    - NumPy
    - Pandas
    - matplotlib
    - scikit-learn
    - scipy
    - numpy
- If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

### 7. Run in Local
- In a terminal or command window, navigate to the top-level project directory customer_segment/ (that contains this README) and run one of the following commands:
```
ipython notebook customer_segment.ipynb

```
```
jupyter notebook customer_segment.ipynb
```
- Notebook will be opened in your default browser.
