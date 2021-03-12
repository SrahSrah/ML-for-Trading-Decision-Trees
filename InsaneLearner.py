import LinRegLearner as lrl 
import BagLearner as bl	  		     			  	   		   	  			  			  		 			     			  	   		   	  			  	
import numpy as np 		  		 			     			  	   		   	  			  	
class InsaneLearner(object): 			  		 			     			  	   		   	  			  			  		 			     			  	   		   	  			  	
    def __init__(self, verbose = False): 			  		 			     			  	   		   	  			  	
		self.n = 20
    def author(self): 			  		 			     			  	   		   	  			  	
        return 'shernandez43' #Georgia Tech username 			  		 			     			  	   		   	  			  				  		 			     			  	   		   	  			  	
    def addEvidence(self, dataX, dataY):
    	self.bagLearners = []
    	for i in range(self.n):
   			learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
   			learner.addEvidence(dataX, dataY)
   			self.bagLearners.append(learner)		
    def query(self, xTest):
    	self.ys = []
    	for i in range(self.n):
    		self.ys.append(self.bagLearners[i].query(xTest))
    	return self.ys
   