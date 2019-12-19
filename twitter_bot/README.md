# Motivation
Create a chatbot that can learn customers questions in the past and response an approriate anwser if the customer asks the similar question in dataset
# Data Source
Collect 8166 twitter questions and their corresponding anwsers of eBay on [Kaggle](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) to feed the model
# Project Description
## Data Preparation

## Word2vec model
 - main_bot.py : run
 
## Demo
# Install Requirements
- Python >= 3.0
- pip install requirement.txt

# Run Chatbot in local
- Check out twitter_bot project
- Create a data folder in twitter_bot
- Download twcs.csv.zip file(https://www.kaggle.com/thoughtvector/customer-support-on-twitter) and put it in data folder
- Go to twitter_bot folder and run command: python main_bot.py
# References
- [gensim](https://radimrehurek.com/gensim/similarities/termsim.html#gensim.similarities.termsim.TermSimilarityIndex): a guideline to build word2vec model, create a matrix of similaries between documents and compute softcosine similarity betwee new document and similiarities matrix
- [cosine similarity](https://www.machinelearningplus.com/nlp/cosine-similarity): an explanation about word2vec and cosine similarity and Euclidean
