""" 			  		 			     			  	   		   	  			  	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			     			  	   		   	  			  	
Atlanta, Georgia 30332 			  		 			     			  	   		   	  			  	
All Rights Reserved 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Template code for CS 4646/7646 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			     			  	   		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			     			  	   		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			     			  	   		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			     			  	   		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			     			  	   		   	  			  	
or edited. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			     			  	   		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			     			  	   		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			     			  	   		   	  			  	
GT honor code violation. 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
-----do not edit anything above this line--- 			  		 			     			  	   		   	  			  	
""" 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
import numpy as np 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
class BagLearner(object): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False): 			  		 			     			  	   		   	  			  	
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.ensemble = []

        for i in range(bags):
            self.ensemble.append(learner(**kwargs)) 

             			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def author(self): 			  		 			     			  	   		   	  			  	
        return 'shernandez43' #Georgia Tech username 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def addEvidence(self,dataX,dataY): 			  		 			     			  	   		   	  			  	
        

        n = len(dataY)
        indexes = range(n)

        for learner in self.ensemble:
            chosenIndexes = np.random.choice(indexes, n, True)
            bagX, bagY = [], []
            for i in chosenIndexes:
                bagX.append(dataX[i])
                bagY.append(dataY[i])
            learner.addEvidence(np.array(bagX), np.array(bagY))


 			  		 			     			  	   		   	  			  	
    def query(self,Xtest): 			  		 			     			  	   		   	  			  	
        
        ensembleYs = []
        for learner in self.ensemble:
            ensembleYs.append(learner.query(Xtest))

        ensembleYs = np.array(ensembleYs)
        ensembleYs = ensembleYs.T

        # Average across each guess:
        avgY = []
        for row in ensembleYs:
            avgY.append(np.mean(row))

        return avgY


 			  		 			     			  	   		   	  			  	
if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    pass
























