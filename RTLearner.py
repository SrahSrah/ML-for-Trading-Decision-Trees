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
from DTLearner import DTLearner as dt                                                                         

 			  		 			     			  	   		   	  			  	
class RTLearner(dt): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def __init__(self, leaf_size = 1, verbose = False): 			  		 			     			  	   		   	  			  	
        dt.__init__(self, leaf_size, verbose)
 			  		 			     			  	   		   	  			  	
    def author(self): 			  		 			     			  	   		   	  			  	
        return 'shernandez43' #Georgia Tech username 

    def getSplitVal(self, dataX, i):
        first = np.random.choice(range(len(dataX)))
        second = np.random.choice(range(len(dataX)))
        return (dataX[first,i] + dataX[second,i])/2			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def determineBestFactor(self, dataX, dataY):
        cols = dataX.shape[1]
        factor = np.random.choice(cols)

        return factor



                                                                       

                                                                                
if __name__=="__main__":                                                                                
    pass











