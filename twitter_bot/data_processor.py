
import pandas as pd
import math
import nlp_helper
from nlp_helper import preprocessing
from nlp_helper import clean_at
import os

class AskeBaytwitter:

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
    
    def clean_data(self):
        self.processed_questions = [ preprocessing(question) for question in self.questions]
        self.processed_responses = [ clean_at(response) for response in self.responses]

    def get_data(self):
        return self.processed_questions, self.processed_responses


    def get_tweet_from_id(self, data, tweet_id):
        """
        @data: pandas dataframe
        @tweet_id: an tweet ID
        @return a tweet corresponding with tweet_id input
        """
        return data[data["tweet_id"] == tweet_id]["text"].values