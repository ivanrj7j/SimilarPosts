import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from random import shuffle

# ------------------ importing the modules ------------------

class Similarity:
    def __init__(self, stopwords='english') -> None:
        self.tfidf = TfidfVectorizer(stop_words=stopwords)
        # initiating the TfidfVectorizer

    def findSimilarPosts(self, allPosts:pd.DataFrame, url:str, totalResults=10, usePoints=True, cap=True):
        if usePoints:
            tfidf_matrix = self.tfidf.fit_transform(allPosts['title']+" "+allPosts['media'], allPosts['points'])
        else:
            tfidf_matrix = self.tfidf.fit_transform(allPosts['title']+" "+allPosts['media'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        # finding the cosine similarities

        indices = pd.Series(allPosts.index, index=allPosts['urlEndPoint'])
        # getting the indices

        index = indices[url]
        # getting the index of the data point  

        sim_scores = enumerate(cosine_similarities[index])
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:totalResults+1] if cap else sim_scores
        sim_indexes = [i[0] for i in sim_scores]
        # getting the scores and the indexes of the top results 

        return allPosts['urlEndPoint'].iloc[sim_indexes]


    def searchPost(self, query:str, posts, totalResults=10):
        allPosts = list(posts.find())
        allPosts.append({'title':query, 'media':'', 'urlEndPoint':'SearchTermFakeID'})
        shuffle(allPosts)
        return self.findSimilarPosts(pd.DataFrame(allPosts), 'SearchTermFakeID', totalResults=totalResults, usePoints=False)