import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import time
from sklearn.metrics.pairwise import linear_kernel

# importing mongodb module

class RecommendPost:
    def __init__(self, collection, projection:dir={}, sortMethod=-1, likeWeight=0.63, commentWeight=1.32, shareWeight=2.7, similarityWeight=1e2, powerWeight=53, vocabularyFile="customWordList.txt") -> None:
        self.collection = collection
        # initializing the collection 

        self.projection = projection
        self.sortMethod = sortMethod
        # initializing all the query filters 

        self.likeWeight = likeWeight
        self.commentWeight = commentWeight
        self.similarityWeight = similarityWeight
        self.powerWeight = powerWeight
        self.shareWeight = shareWeight
        self.stemmer = PorterStemmer()
        # initializing the weights 

        with open(vocabularyFile, 'r') as file:
            words = file.read()
        self.tfidfVectorizer = TfidfVectorizer()
        self.tfidfVectorizer.fit([words])
        # initializing the vectorizer 

    def calculatePower(self, data:pd.DataFrame):
        nonNormalizedPower = (data['points']*self.likeWeight) + (data['comments']*self.commentWeight) + (data['shares']*self.shareWeight)
        return nonNormalizedPower / data['points'].max()

    def score(self, data:pd.DataFrame):
        return (data['similarity'] * self.similarityWeight) + (data['power'] * self.powerWeight)

    def similar(self, data:pd.DataFrame, contentVector, queryVector):
        data['similarity'] = linear_kernel(contentVector, queryVector)
        data['score'] = self.score(data)
        data = data.sort_values('score', ascending=False)
        
        return data

    def similarPost(self, url, returnUrlOnly=True, topN=True, top=100, **kwargs):
        postContent = dict(self.collection.find_one({'urlEndPoint':url}, projection={"title":True, "media":True, "_id":False}))
        postContent = postContent['title'] + " " + postContent['media']
        # getting the content of the post  
        
        data = pd.DataFrame(list(self.collection.find({'$text':{'$search':postContent},}, projection=self.projection, sort=[('points',-1)],limit=3070)))
        # getting the data 

        

        data['power'] = self.calculatePower(data)
        # calculating the power 

        
        contentVector = self.tfidfVectorizer.transform(data['title'] + " " + data['media'])        
        queryVector = self.tfidfVectorizer.transform([postContent])
        # getting the index of the query and creating vector for content and query         
        
        data = self.similar(data, contentVector, queryVector)[1:top+1] if topN else self.similar(data, contentVector, queryVector)[1:]
        # finding similar 
               
        return data['urlEndPoint'] if returnUrlOnly else data

