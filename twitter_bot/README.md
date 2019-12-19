# Motivation
Create a chatbot that can learn customers questions in the past and response an approriate anwser if the customer asks the similar question in dataset
# Data Source
Collect 8166 twitter questions and their corresponding anwsers of eBay on [Kaggle](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) to feed the model
# Project Description
## Data Preparation
- Filter all questions of eBay's customers and the answers from supporters in twitter customer support dataset
````
  def __init__(self, datapath, fileName):
     self.datapath = datapath
     self.fileName = fileName
     self.questions = []
     self.responses = []
     self.processed_questions = []
     self.processed_responses = []
     self.author_name = "AskeBay"

def read_data(self):
    data = pd.read_csv(self.datapath + os.path.sep + self.fileName)
    # select questions from ebay's customers
    df_questions = data[ data["text"].str.contains(self.author_name) ]
    # select all support messages of eBay
    #df_ebay = data[ data['author_id'] == self.author_name ] 
    # We reselect the necessary columns:
    # tweet_id, text, response_tweet_id, in_response_to_tweet_id
    #in_response_to_tweet_id
    select_features = ["tweet_id", "text",	"response_tweet_id", "in_response_to_tweet_id"]
    df_questions = df_questions[select_features]
    for i in range(len(df_questions)):

        tweet = df_questions.iloc[i]
        text = tweet["text"]
        self.questions.append(text)

        rep_ids = tweet["response_tweet_id"]
        if (rep_ids and (not pd.isna(rep_ids))):
            # if more than one answer for one question, pick the first one
            if "," in rep_ids:
                rep_ids = rep_ids.split(",")[0]
            response = "".join(self.get_tweet_from_id(data, int(rep_ids)))
            self.responses.append(response)
        else:
                self.responses.append("")

    self.clean_data()
````
- Data cleaning: remove punctuation and stop words, do the lemmatization and decontraction. Finally , sentences are tokenized before being transformed into model
```
def preprocessing(text):
 text = clean_emoji(text)
 text = decontracted(text)
 text = clean_punctuation(text)
 text = clean_num(text)
 text = normalize(text)
 return text
````

## Word2vec Model and Similarity Matrix
- Initialize [Dicionary](https://radimrehurek.com/gensim/corpora/dictionary.html) from corpus of clean and tokenized questions. It maps each word with an unique id.
- Transform dictionary into [TF-IDF model](https://radimrehurek.com/gensim/models/tfidfmodel.html). The purpose is to reduce the weight of words with high frequence in corpus.
- Create worrd2vec model from questions corpus and build a sparse term similarity matrix using a term similarity index
- Create bag-of-word matrix from corpus and meantime, transform it into TF-TDF matrix
- Compute the distance between documents in corpus with Soft Cosine Similarities approach
```
def compute_sim_matrix(self):
    '''    
    if(self.model_type.lower() == "fasttext"):
        model = FastText(self.questions) 
    else:
        model = Word2Vec(self.questions)
    '''
    self.dictionary = Dictionary(self.questions)
    self.tfidf = TfidfModel(dictionary = self.dictionary)
    word2vec_model = Word2Vec(self.questions
                        , workers=cpu_count()
                        , min_count=5
                        , size=300
                        , seed=12345)

    sim_index = WordEmbeddingSimilarityIndex(word2vec_model.wv)
    sim_matrix = SparseTermSimilarityMatrix(sim_index
                                                , self.dictionary
                                                , self.tfidf
                                                , nonzero_limit=100)
    bow_corpus = [self.dictionary.doc2bow(document) for document in self.questions]

    tfidf_corpus = [self.tfidf[bow] for bow in bow_corpus]

    self.docsim_index = SoftCosineSimilarity(tfidf_corpus, sim_matrix, num_best=10)
```
 
## Demo
# Install Requirements
- Python >= 3.0
- pip install requirement.txt

# Run Chatbot in local
- Check out twitter_bot project
- Create a data folder in twitter_bot
- Download [twcs.csv.zip file](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) and put it in data folder
- Go to twitter_bot folder and run command: <code>python main_bot.py</code>
# References
- [gensim](https://radimrehurek.com/gensim/similarities/termsim.html#gensim.similarities.termsim.TermSimilarityIndex): a guideline to build word2vec model, create a matrix of similaries between documents and compute softcosine similarity betwee new document and similiarities matrix
- [cosine similarity](https://www.machinelearningplus.com/nlp/cosine-similarity): an explanation about word2vec and cosine similarity and Euclidean
