



### Motivation
An investment may make sense if we expect it to return more money than it costs. But returns are only part of the story because they are risky - there may be a range of possible outcomes. How does one compare different investments that may deliver similar results on average, but exhibit different levels of risks? The Sharpe ratio has been one of the most popular risk/return measures in finance because it is simple and easy to use. 

### Project Description
In this project, Principal Component Analysis is used to structure porfolio returns of 20 stocks of S&P 500 index. First step is to collect historical finance data in 5 years from https://www.alphavantage.co. Next step is to calulate anualized returns of 20 stocks, normalizing them and to fit it to PCA. Finnaly, we extract eigen porfolios from variances of PCA components and find the porfolio with the highest sharp ratio

### Data Collection
- To collect stock time series data from https://www.alphavantage.co, we call parameterized APIs and recieve csv with requested prices in 20 years
````
def get_raw_data(ticker):
    """Return pandas dataframe given ticker symbol.
       dataframe has 8 columns: open, high, low, close, adjusted_close, volume and split_coefficient
    """
    query =  "https://www.alphavantage.co/query?"
    function = "function=TIME_SERIES_DAILY"
    symbol = "symbol=" + ticker
    key = "apikey=6PI9I1HMXX921ZEP"
    dtype =  "datatype=csv"
    size = "outputsize=full"
    
    url = query + "&" + function + "&" + symbol + "&" + key + "&" + dtype + "&" + size
    print(url)
    
    df =pd.read_csv(url)
    #print(df.head())
    df = df.set_index(['timestamp'])
    #rename index column from timestame to Date
    df.index.names = ['Date']

````
- The API for one stock will look like: 
https://www.alphavantage.co/query?&function=TIME_SERIES_DAILY&symbol=MSFT&apikey=6PI9I1HMXX921ZEP&datatype=csv&outputsize=full
- It is about 20 csv files in return after sendind 20 APIs to provider. In the project's scope, we work only with closing prices in 5 years. We select 2 columns : close and date and create a new dataframe with them:
````
def get_data(tickers, start_date, end_date):
  
  dates = pd.date_range(start_date, end_date)
  df = pd.DataFrame(index = dates)
  df.index.names = ["Date"]
  
  for ticker in tickers:


    try:
      df_temp = get_raw_data(ticker)
      #print(df_temp.head())
      #print(ticker)
      # select close column only and rename it
      df_temp = df_temp[['close']].rename(columns={'close': ticker})
      df = df.join(df_temp)
    except:
      print("Error while downloading: " + ticker)
            
  return df.dropna()
````
- New dataframe with 5 first rows:
![](/stock_analysis/images/5rows.PNG)

- Plot cumulative returns to observe the total change in price over 5 years:
````
(asset_prices/asset_prices.iloc[0]).plot(figsize=(15, 8))

````
![](/stock_analysis/images/cumulative_returns.PNG)

### Asset Returns Calculation & Standardization

- The first step is to calculate percent returns : (price(t) - price(t-1))/price(1)
- Standardizing percent returns so that all the variables will be transformed to the same scale: (price - price.mean())/price.std())
````
def calculate_returns(stocks_prices):
    """
    - stock_prices: is a dataframe whose columns represent stock names and rows are date
    
    - return daily stocks returned after they are normalized
    
    """
    # Percentage of returns = (price(t) - price(t-1))/price(1)
    stock_returns = stocks_prices.pct_change().dropna()
    normalized_returns = (stock_returns - stock_returns.mean())/stock_returns.std()
    
    return stock_returns, normalized_returns

stock_returns, normalized_returns = calculate_returns(asset_prices)

````
### Machine Learning Model
#### Train/Test Split
- Normalized return stocks is splitted into 80% of data for train data and 20% for test
- Train set will be fitted to PCA in order to extract the variance from features
- Test set will be used to find the best eigen porfolio later
````
train_size = int(len(normalized_returns) * 0.8)

train = normalized_returns[:train_size]
test = normalized_returns[train_size:]

train_raw = stock_returns[:train_size]
test_raw = stock_returns[train_size:]
````
#### PCA Model
- Calculate covariance matrix from training data set and fit it to PCA model
- Find how many PCA components can be kept to to gain 90% variance
````
# Taking out SPX
stock_tickers = stock_returns.columns.values[:-1]
cov_matrix = train[stock_tickers].cov()
pca = PCA()
pca.fit_transform(cov_matrix)
var_threshold = 0.9
# Calculate the cummulative of explained variance ratio of all PCA Components
var_explained_cum = np.cumsum(pca.explained_variance_ratio_)
n_comp = np.where(np.logical_not(var_explained_cum < var_threshold))[0][0] + 1  # +1 due to zero based-arrays
print('%d components explain %.2f%% of variance' %(n_comp, 100* var_threshold))
````
- Print() results 9 components and we loose about 10% information
- Visualize PCA explained variance and explained variance ratios of first 9 components:
````
# Plot PCA explained variances ratio

index = ['PC {}'.format(ix+1) for ix in range(n_comp)]
df_var_explained_ratio = pd.DataFrame({"Explained Variance Ratio": pca.explained_variance_ratio_[:n_comp]*100}
                                      , index = index)
# Create a bar plot visualization
df_var_explained_ratio.plot(kind="bar"
                            , color="purple"
                            , title="Explained Variance Ratio by PCA")
df_var_explained_cum = pd.DataFrame({"Explained Variance Ratio Cummulative": var_explained_cum[:n_comp]}
                                    , index = index)
df_var_explained_cum.plot()
df_var_explained = pd.DataFrame({"Explained Variance": pca.explained_variance_[:n_comp]*100}
                                      , index = index)
df_var_explained.plot()
````
![](/stock_analysis/images/explained_variance_ratio.PNG)![](/stock_analysis/images/cum_explained_variance_ratio.PNG)![](/stock_analysis/images/explained_variance.PNG)
    
- <b>Observation</b>: As you we can see, the very first bar represented in the first principal component is the highest one. Taken alone, it explains about 38% of the total variance of the stock returns in the index. The rest of eigenvalues are much smaller than the first one, and explain much smaller fraction of the total variance.

- Important Features: Find the feature whose variance is the highest in each pca components:
````
n_pcs = pca.components_.shape[0]
important_idx = [np.argmax(pca.components_[i]) for i in range(n_pcs)]
important_features = [stock_tickers[important_idx[i]] for i in range(n_pcs)]
pd.DataFrame({"PC {}".format(i+1): [important_features[i]] for i in range(n_pcs)}, index = ["Important Feature"]).T
````
![](/stock_analysis/images/important_feature.PNG)
    
### Eigen Porfolio Returns Calculation
- Each PCA component contains the eigen weight of stocks. It can be represented in a matrix whose rows are variances and columns are tickers. The result is acquired by 3 steps:
- We will iterate pca.components and compute the weight of return of each stock:
````
def eigen_porfolio_return(idx, pca, stock_tickers):
    """
    - @idx: index of pca components (idx th PCA components)
    - @pca: PCA()
    - @stock_tickers: is an array - a list of stock name
    - return a pandas frame whose data is weight of stock derived from idx th PCA components
    """
    pcs = pca.components_
    # Normalize porfolio to 1
    # Porfolio's weights is an array
    pc_w = pcs[:, idx ]/sum(pcs[:, idx])
    eigen_prtf = pd.DataFrame(data ={'weights': pc_w}
                              , index = stock_tickers)
    eigen_prtf.sort_values(by=['weights']
                           , ascending=False
                           , inplace=True)
    return eigen_prtf
 ````
 - Caculate the sharp-ratio from porfolio returns:
 ````
 def sharpe_ratio(eigen_prtf_returns, periods_per_year=252):
    """
    sharpe_ratio - Calculates annualized return, annualized vol, and annualized sharpe ratio, 
                    where sharpe ratio is defined as annualized return divided by annualized volatility 
                    
    @eigen_prtf_returns - pd.Series of returns of a single eigen portfolio
    @Return: a tuple of three : annualized return, volatility, and sharpe ratio
    """
    annualized_return = 0.
    annualized_vol = 0.
    annualized_sharpe = 0.
    # compute the number of years from given period 252 which is the number of business days per year
    # take the number of porfolio and divide to given period
    n_years = len(eigen_prtf_returns)/periods_per_year
    # Cummulative returns
    annualized_return = np.power(np.prod(1 + eigen_prtf_returns), (1/n_years) )-1
    annualized_vol = eigen_prtf_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol
    return annualized_return, annualized_vol, annualized_sharpe
 ````
 - Finally, combine step 1 and step 2 to figure out risk-free porfolio:
 ````
 def optimize_porfolio(pca, cov_matrix, stock_tickers, stock_return_data):
    """
    pca: Principal Components after fitting data
    cov_matrix: covariance matrix of normalized stock returns
    stock_tickers: an array of stock names
    stock_return_data: an asset prices dataframe whose columns are stock names and rows are time
    """
    pcs = pca.components_
    n_portfolios = len(pcs) 
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    # n_portfolios = len(pcs)
    for ix in range(n_portfolios):
        # extract eigen value of features in each pca components 
        eigen_prtf = eigen_porfolio_return(ix, pca, stock_tickers)
        # Multiply stocks (stock_return_data) with their eigen portfolio (eigen_prtf)  in corresponding
        eigen_prtf_returns = np.dot(stock_return_data[stock_tickers], eigen_prtf)
        eigen_prtf_returns = pd.Series(eigen_prtf_returns.squeeze()
                                       , index=stock_return_data.index)
        eigen_prtf_returns = eigen_prtf_returns.dropna()
        ret, vol, sharpe = sharpe_ratio(eigen_prtf_returns)
        annualized_ret[ix] = ret
        sharpe_metric[ix] = sharpe 
        annualized_vol[ix] = vol
    return annualized_ret, annualized_vol, sharpe_metric
 
 ````
 - Using optimize_porfolio() with testset, We got : Eigen portfolio #4 BRK.B with the highest Sharpe. Return = 32.35, Volatility = 46.24, Sharpe = 0.70
 - Plot the most optimized porfolio:
 ![](/stock_analysis/images/optimized_porfolio.PNG)
 
### Installation & Software Requirements

#### Requirements
- Python >= 3.7
- Jupyter Notebook

#### Dependencies
- pandas
- matplotlib
- seaborn
- scipy
- requests
- numpy
- sklearn

### Run Notebook in the local
- checkout the project : git clone https://github.com/diem-ai/Risk-and-Return-ROI.git
- In a terminal or command window, navigate to the top-level project directory sentiment_mining/ (that contains this README) and run one of the following commands:
    ````
    ipython notebook sentiment_prediction2.ipynb
    ````

### Run the Notebook in Google Colab
- Click on the link below to run the notebook on Colab

https://colab.research.google.com/drive/1RG5PlDLkiaZHtFQuew5iJfsV5yX6taNB#scrollTo=x5yOIQMG1SsN&uniqifier=10







