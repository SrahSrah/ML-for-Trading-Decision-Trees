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
import pandas as pd	  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
class DTLearner(object): 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def __init__(self, leaf_size = 1, verbose = False): 
        # initialize our tree
        # final tree will be of form #Nodes x 3
        # columns will be of form [Best Factor, splitVal, Left Tree, Right Tree]			  		 			     			  	   		   	  			  	
        self.leaf_size = leaf_size	
        self.tree = [] 
        self.treeBuilt = False 		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    def author(self): 			  		 			     			  	   		   	  			  	
        return 'shernandez43' #Georgia Tech username 			  		 			     			  	   		   	  			  	
 	
    def determineBestFactor(self, dataX, dataY):        

        if len(dataX.shape) != 1:
            data = np.concatenate((dataX,dataY[:,None]), axis = 1)
        else: 
            data = np.concatenate((dataX[:,None], dataY[:,None]), axis = 1)
        

        corells = np.corrcoef(dataX, dataY[:,None], rowvar = False)
        

        return np.nanargmax(np.abs(corells[:-1, -1]))

    def getSplitVal(self, dataX, i):
        return np.median(dataX[:, i])

    def buildTree(self, dataX, dataY):

        #Stop Criteria 1: if the rows of data left is less than or equal to the leaf size:
        if dataY.shape[0] <= self.leaf_size:
            return [-1, float(np.mean(dataY)), -1, -1]

        #Stop Criteria 2: if all remaining y's are the same, return a leaf
        y = float(dataY[0])
        
        if (dataY == y).all(): #dataY.shape[0] == testY.shape[0]:
            return [-1, y, -1, -1]                

        # Else: Build more tree:
        else: 
            i = self.determineBestFactor(dataX, dataY)
            splitVal = self.getSplitVal(dataX, i)

            newXL = dataX[dataX[:,i] <= splitVal,:]
            newYL = dataY[dataX[:,i] <= splitVal]
            newXR = dataX[dataX[:,i] > splitVal, :]
            newYR = dataY[dataX[:,i] > splitVal]
            
            # If all of the data is shuffled to one side due to the choice of median, create a leaf:
            if newXL.size == 0 or newXR.size == 0:
                return [-1, float(np.mean(dataY)), -1, -1]
              

            # If not, keep building the tree:
            leftTree = self.buildTree(newXL, newYL)
            rightTree = self.buildTree(newXR, newYR)

            checkLeft = np.array(leftTree)
            if len(np.shape(checkLeft)) == 1:
                root = [i, splitVal, 1, 2]
            else:
                root = [i, splitVal, 1, len(leftTree)+1]
          
            return (np.vstack([root,leftTree, rightTree]))


    def addEvidence(self,dataX,dataY): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Add training data to learner. Here, we're building the data tree 			  		 			     			  	   		   	  			  	
        @param dataX: X values of data to add 			  		 			     			  	   		   	  			  	
        @param dataY: the Y training values 			  		 			     			  	   		   	  			  	
        """ 	

        # convert data to np if not already:    
        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # Build the tree        
        self.tree = self.buildTree(dataX, dataY)
        self.treeBuilt = True
        
 
 			  		 			     			  	   		   	  			  	
    def query(self,Xtest): 			  		 			     			  	   		   	  			  	
        """ 			  		 			     			  	   		   	  			  	
        @summary: Estimate a set of test samples given the model we built. 			  		 			     			  	   		   	  			  	
        @param Xtest: should be a numpy array with each row corresponding to a specific query. 			  		 			     			  	   		   	  			  	
        @returns the estimated values according to the saved model. 			  		 			     			  	   		   	  			  	
        """ 	
        y = []

        tree = self.tree

        for x in range(Xtest.shape[0]):
            index = 0
            yFound = False
            while not yFound: #tree[index][0] != -1:

                factor = int(tree[index][0])
                splitVal = tree[index][1]
                if Xtest[x][factor] <= splitVal:                    # Go Left
                    leftIndex = int(tree[index][2])
                    if leftIndex == -1:                             # If we reach a leaf on the left,
                        y.append(tree[index][1])                    # Append predicted y value to y array
                        yFound = True
                    else:                                           # Else, add to index and keep traversing tree
                        index += leftIndex
                else:                                               # Go Right
                    rightIndex = int(tree[index][3])
                    if rightIndex == -1:                            # If we reach a leaf on the right,
                        y.append(tree[index][1])                    # Appeand that leaf
                        yFound = True
                    else:
                        index += rightIndex                         # Else, keep traversing tree

        return y			



 			  		 			     			  	   		   	  			  	
if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    pass

