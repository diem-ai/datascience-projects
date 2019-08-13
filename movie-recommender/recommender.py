import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from ast import literal_eval
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string


class Similarity:

    def make_recommendation(idx_movie
                        , latent_features
                        , movies
                        , printed_features
                        , top_n):

        """
        @movie_id: movie id
        @sim_matrix: matrix factorization (either item rating matrix or content based matrix)
        @movies: pandas dataframe
        @printed_featers: list of features shown in final result
        @top_n: the number of movies came out
        @Return similarity scores between the seed_movie and all other movies. 

        """ 
        #avoid exception if movie is not in the training set
    #get style variant's feature vector in latent space

        item_vector = np.array(latent_features.loc[idx_movie]).reshape(1, -1)

        #calculate similarity

        similarities = cosine_similarity(latent_features, item_vector,  dense_output=True)

        # get detailed movie info

        similarities = pd.DataFrame(similarities, index = latent_features.index.tolist())

        similarities.columns = ['score']

        similarities['score'] = similarities.apply(lambda x : np.round(x, 4))

        sim_df = pd.merge(movies, similarities, left_index=True, right_index=True)

        sim_df.sort_values('score', ascending=False, inplace=True)

        print(sim_df[printed_features][1:top_n])

        """
        try:
            item_vector = np.array(sim_matrix.loc[movie_id]).reshape(1, -1)
            #calculate similarity
            similarities = cosine_similarity(sim_matrix, item_vector, dense_output=True)
            # get detailed movie info
            similarities = pd.DataFrame(similarities, index = sim_matrix.index.tolist())
            similarities.columns = ['sim_score']
            sim_df = pd.merge(movies, similarities, left_index=True, right_index=True)
            #sim_df['sim_score'] = sim_df['sim_score'].apply(np.round(4))
            sim_df.sort_values('sim_score', ascending=False, inplace=True)
            return sim_df
            ##print(sim_df.head(top_n))
            #print_recommendation(sim_df, top_n)
        except:
            print("Your movie is not in the training set. Please make another recommendation")
            return 
        """

    def print_recommendation(similarity_df, top_n):
        similarity_df['sim_score'] = np.round(4)
        similarity_df.sort_values('sim_score', ascending=False, inplace=True)
        print(similarity_df.head(top_n))
        






class Utility:

    def print_msg(message):
        print(message)
    
    def print_another_msg(message):
        print("another message")
        print(message)

    def get_movie_by_title(movies, title):
        """
        @movies: pandas dataframe that contains title and id column 
        @tile: movie name
        @return: an interger
        """
        item = movies[movies['title'] == title]['id']
        if (len(item)) == 0:
            return -1
        else:
            return item.to_numpy()[0]

    def get_idx_by_title(movies, title):
        """
        @movies: pandas dataframe that contains title and id column 
        @tile: movie name
        @return: an interger
        """
        item = movies[movies['title'] == title]
        if (len(item)) == 0:
            return -1
        else:
            return item.index.tolist()[0]
        # 
        #         return movies[movies['title'] == title]['id']
    # Function that computes the weighted rating of each movie
    def weighted_rating(self, data, m, C):
        """
        data: pandas dataframe
        m: the minimum number of votes required to be in the chart
        C: mean of vote average
        """
        v = data['vote_count']
        R = data['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    def get_top_n_movies(data, top_n):
        """
        data: Pandas dataframe that contains "score" column as rating
        top_n: an integer value
        return datafram with top_n movies with highest score
        """
        data.sort_values('vote_average', ascending=False)
        return data.head(top_n)

    def get_genre(genre_list):
        """
        @genre_list: list of dictionary objects under json format
        @genre_list = [{'id': 10749, 'name': 'Romance'}, {'id': 35, 'name': 'Comedy'}]
        
        return: string: Romance Comedy
        """
        names = [''.join(item['name'].lower()) for item in genre_list]
        return ' '.join(names)
    
    def get_cast(casts):
        names = [''.join(item['name'].replace(' ','').lower()) for item in casts]
        return ' '.join(names)  

        
    def get_wordnet_pos(treebank_tag):
        """Convert the part-of-speech naming scheme
        from the nltk default to that which is
        recognized by the WordNet lemmatizer"""

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    
    def preprocess_series_text(data):
        """Perform complete preprocessing on a Pandas series
        including removal of alpha numerical words, normalization,
        punctuation removal, tokenization, stop word removal, 
        and lemmatization."""
        # remove alpha numerical characters
        numeric_re = re.compile('^[0-9]+$')
        numeric_lambda = lambda x: numeric_re.sub('', x)
        data = data.map(numeric_lambda)

        # remove alpha numerical words and make lowercase
        alphanum_re = re.compile(r"""\w*\d\w*""")
        alphanum_lambda = lambda x: alphanum_re.sub('', x.strip().lower())

        data = data.map(alphanum_lambda)

        # remove punctuation
        punc_re = re.compile('[%s]' % re.escape(string.punctuation))
        punc_lambda = lambda x: punc_re.sub(' ', x)

        data = data.map(punc_lambda)
        # tokenize words
        data = data.map(word_tokenize)

        # remove stop words
        sw = stopwords.words('english')
        sw_lambda = lambda x: list(filter(lambda y: y not in sw, x))

        data = data.map(sw_lambda)

        # part of speech tagging--must convert to format used by lemmatizer
        data = data.map(nltk.pos_tag)
        pos_lambda = lambda x: [(y[0], Utility.get_wordnet_pos(y[1])) for y in x]
        data = data.map(pos_lambda)

        # lemmatization
        lemmatizer = WordNetLemmatizer()
        lem_lambda = lambda x: [lemmatizer.lemmatize(*y) for y in x]
        data = data.map(lem_lambda)
        
        return data.map(' '.join)

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''



if __name__ == "__main__":
    
    Utility.print_msg("hahaha")
    path = os.getcwd() + os.path.sep + "data" + os.path.sep 

    metadata_file = path + "small_movies_metadata.csv"
    movies = pd.read_csv(metadata_file, low_memory=False)
    movies['genres'] = movies['genres'].apply(lambda x : literal_eval(x))
    movies['genres'] = movies['genres'].apply(lambda x : Utility.get_cast(x))
    print(movies['genres'][:2])
    
    #rating_matrix[278]
    seed_movie = Utility.get_movie_by_title(movies, "Dracula: Dead and Loving It")
    #seed_movie = seed_movie.to_numpy()
    #seed_movie = seed_movie[0]
    #print(seed_movie.to_numpy()[0])
  
    print("movie_id: {}".format(seed_movie))
    latent_matrix_df = pd.read_csv(os.getcwd() + os.path.sep + "data" + os.path.sep + "latent_matrix.csv"
                                    , index_col=0) 
    #print(latent_matrix_df.head())
    features = ['title', 'genres','vote_count', 'vote_average','score']
    Similarity.make_recommendation(710, latent_matrix_df, movies, features, 10)
    #print(len(df))


