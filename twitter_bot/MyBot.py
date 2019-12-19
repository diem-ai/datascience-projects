"""
Class SaleBot
It is initialised by nlp model (bag-of-word, tf-idf, word2vec)
It returns response with a question as the input
"""
from gensim.corpora import Dictionary
#from gensim.models import FastText
from gensim.models import Word2Vec , WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from gensim.models import TfidfModel
from multiprocessing import cpu_count
from nlp_helper import preprocessing


class AskeBayBot:
    """
    - Using tf-idf and word2vec to build  vector matrix from the corpus
    - Using soft-cosine similarity to calculate the similarity between query and matrix
    """
    """
    References:
    - https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb

    """

    def __init__(self, questions, responses, model_type="word2vec"):
        self.questions = questions
        self.responses = responses
        self.model_type = model_type
        self.docsim_index = []
        self.dictionary = []
        self.tfidf = []
        self.compute_sim_matrix()
    
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

    def get_similarities(self, question):
        '''
        @return indices of anwsers whose questions are similar to the input question
        '''
        vectorizer = self.dictionary.doc2bow(preprocessing(question))
        tfidf_vectorizer = self.tfidf[vectorizer]
        similarities = self.docsim_index[tfidf_vectorizer]
        return similarities

    def get_response(self, question):
        similarities = self.get_similarities(question)
        return self.get_sim(similarities, 1)
    
    def get_all_responses(self, question):
        similarities = self.get_similarities(question)
        return self.get_sim(similarities, 10)
    
    def get_sim(self, similarities, n_top=1):
        """
        @return a tuple of similar question and best response in similarity matrix
        """
        sim_questions = []
        sim_responses = []
        sim_scores = []
        
        if (len(similarities) > 0):
            for (idx, score) in similarities:
                if (idx < len(self.responses)):
                    sim_questions.append(self.questions[idx])
                    sim_responses.append(self.responses[idx])
                    sim_scores.append(score)
                    # return self.questions[idx], self.responses[idx], score
                
        else:
            return "Just a moment, someone will contact you"

        if (n_top == 1):
            return sim_questions[0], sim_responses[0], sim_scores[0]
        else:
            return sim_questions, sim_responses, sim_scores



if __name__ == "__main__":
    print("I'm a bot")