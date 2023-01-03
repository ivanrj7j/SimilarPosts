import pandas as pd

class RecommendPosts:
    def __init__(self, postsCollection, similarityModule) -> None:
        self.postsCollection = postsCollection
        self.similarityModule = similarityModule

    def findSimilar(self, queryPostUrl, totalResults=10, usePoints=True, cap=True):
        queryPost = dict(self.postsCollection.find_one({'urlEndPoint':queryPostUrl}))
        content = queryPost['title'] + ' ' + queryPost['media']
        # getting the content to search 

        posts = list((self.postsCollection.find({'$text':{'$search':content}}).sort('points', -1)))
        
        if len(posts) >= 3070:
            posts = posts[:3070]
        # getting all the posts, stripping if its too long

        if queryPost not in posts:
            posts.append(queryPost)
        # appending the query if its not in the posts list for some reason 

        return self.similarityModule.findSimilarPosts(pd.DataFrame(posts), queryPostUrl, totalResults, usePoints, cap)
