""" 			  		 			     			  	   		   	  			  	
Test a learner.  (c) 2015 Tucker Balch 			  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
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
import math 			  		 			     			  	   		   	  			  	
import LinRegLearner as lrl 
import InsaneLearner as it
import RTLearner as rt
import DTLearner as dt	
import BagLearner as bl	  		 			     			  	   		   	  			  	
import sys
import util    
import matplotlib.pyplot as plt  
import time                                                                           

class tester():

    def __init__(self, data):
        np.random.shuffle(data)
        # compute how much of the data is training and testing                                                                              
        train_rows = int(0.6* data.shape[0])                                                                                
        test_rows = data.shape[0] - train_rows                                                                          
                                                  
        # separate out training and testing data                                                                                
        self.trainX = data[:train_rows,0:-1]                                                                                 
        self.trainY = data[:train_rows,-1]                                                                               
        self.testX = data[train_rows:,0:-1]                                                                              
        self.testY = data[train_rows:,-1]                

    def question1(self):

        inSampleErr, outSampleErr = [], []

        for i in range(len(self.trainX)):
            learner = dt.DTLearner(i, False)
            learner.addEvidence(self.trainX, self.trainY)                                                                              
            inErr, outErr = self.getRSME(learner, verbose = False)
            inSampleErr.append(inErr)
            outSampleErr.append(outErr)

        plt.plot(inSampleErr, label = "In-Sample RSME")
        plt.plot(outSampleErr, label = "Out-Sample RSME")
        plt.xlabel("Leaf Size")
        plt.ylabel("RSME")
        plt.title("Figure 1: \n Leaf Size effect on In-Sample and Out-Sample RSME")
        plt.legend()
        plt.savefig("Figure 1.png")


        plt.figure()
        plt.plot(inSampleErr, label = "In-Sample RSME")
        plt.plot(outSampleErr, label = "Out-Sample RSME")
        plt.xlabel("Leaf Size")
        plt.ylabel("RSME")
        plt.title("Figure 2: \n Leaf Size effect on In-Sample and Out-Sample RSME")
        plt.legend()
        plt.xlim(0,100)
        plt.savefig("Figure 2.png")



        return inSampleErr, outSampleErr

    def question2(self, noBagInErr, noBagOutErr):

        avgBagInRSME, avgBagOutRSME = [], []
        for j in range(5):

            bagInSampleErr, bagOutSampleErr = [], []
            for i in range(len(self.trainX)):
                learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size" : i, "verbose" : False}, bags = 15, boost = False, verbose = False)
                learner.addEvidence(self.trainX, self.trainY)                                                                              
                inErr, outErr = self.getRSME(learner, verbose = False)
                bagInSampleErr.append(inErr)
                bagOutSampleErr.append(outErr)

            avgBagInRSME.append(bagInSampleErr)
            avgBagOutRSME.append(bagOutSampleErr)


        avgBagInRSME = np.array(avgBagInRSME)
        avgBagOutRSME = np.array(avgBagOutRSME)

        avgBagInRSME = np.mean(avgBagInRSME, axis = 0)
        avgBagOutRSME = np.mean(avgBagOutRSME, axis = 0)

        plt.figure() 
        plt.plot(noBagInErr, label = "Non-Bagged In-Sample RSME")
        plt.plot(noBagOutErr, label = "Non-Bagged Out-Sample RSME")
        plt.plot(avgBagInRSME, label = "Bagged In-Sample RSME")
        plt.plot(avgBagOutRSME, label = "Bagged Out-Sample RSME")
        plt.xlabel("Leaf Size")
        plt.ylabel("RSME")
        plt.title("Figure 3: \n Leaf Size effect on In-Sample and Out-Sample RSME: \n Bagged vs. Non-Bagged Results")
        plt.legend()
        plt.savefig("Figure 3.png")

        plt.figure()
        plt.plot(noBagInErr, label = "Non-Bagged In-Sample RSME")
        plt.plot(noBagOutErr, label = "Non-Bagged Out-Sample RSME")
        plt.plot(avgBagInRSME, label = "Bagged In-Sample RSME")
        plt.plot(avgBagOutRSME, label = "Bagged Out-Sample RSME")
        plt.xlabel("Leaf Size")
        plt.ylabel("RSME")
        plt.title("Figure 4: \n Leaf Size effect on In-Sample and Out-Sample RSME: \n Bagged vs. Non-Bagged Results")
        plt.legend()
        plt.xlim(0,100)
        plt.savefig("Figure 4.png")



    def question3(self, verbose = False):

        # Compare DT and RT Learners:

        dtLearner = dt.DTLearner(15, False)
        dtLearner.addEvidence(self.trainX, self.trainY)                                                                              

        rtLearner = rt.RTLearner(15, False)
        rtLearner.addEvidence(self.trainX, self.trainY)                                                                              


        dtINcorr, dtOUTcorr, rtINcorr, rtOUTcorr = [], [], [], []   # Store average correlations
        dtTime, rtTime = [], []                                     # Store average time to build tree
        dtQTime, rtQTime = [], []                                   # Store average time to query tree

        for i in range(5):
            #DT
            dtLearner = dt.DTLearner(15, False)
            
            start = time.clock()
            dtLearner.addEvidence(self.trainX, self.trainY) 
            dtTime.append(time.clock() - start)                                                                             
            
            dtIn, dtOut = self.getCorr(dtLearner, True)
            dtINcorr.append(dtIn)
            dtOUTcorr.append(dtOut)
            
            start = time.clock()
            _ = dtLearner.query(self.testX)
            dtQTime.append(time.clock() - start)

            # RT
            rtLearner = rt.RTLearner(15, False)
            
            start = time.clock()
            rtLearner.addEvidence(self.trainX, self.trainY) 
            rtTime.append(time.clock() - start)         
            
            rtIn, rtOut = self.getCorr(rtLearner, True)
            rtINcorr.append(rtIn)
            rtOUTcorr.append(rtOut)
            
            start = time.clock()
            _ = rtLearner.query(self.testX)
            rtQTime.append(time.clock() - start)
            
        # Calculate means of each testing parameter:
        dtINcorr = np.mean(dtINcorr)
        dtOUTcorr = np.mean(dtOUTcorr)
        rtINcorr = np.mean(rtINcorr)
        rtOUTcorr = np.mean(rtOUTcorr)

        
        dtTime = np.mean(dtTime)
        rtTime = np.mean(rtTime)

        dtQTime = np.mean(dtQTime)
        rtQTime = np.mean(rtQTime)

        # Metric 1: Average Correlation:
        plt.figure()
        ngroups = 2
        dtCorr = [dtINcorr, dtOUTcorr]
        rtCorr = [rtINcorr, rtOUTcorr]

        fig, ax = plt.subplots()
        index = np.arange(ngroups)
        bar_width = 0.35
        rects1 = plt.bar(index, dtCorr, bar_width, label = "Classic Decision Tree")
        rects2 = plt.bar(index+bar_width, rtCorr, bar_width, label = "Random Tree")

        plt.xlabel("Type of Data")
        plt.xticks(index + bar_width/2, ["In-Sample", "Out-Sample"])
        plt.legend(loc = 5)
        plt.ylabel("Correlation")
        plt.title("Figure 5: \n Average Correlation: Classic Decision Trees vs. Random Trees")
        plt.savefig("Figure 5.png")


        # Metric 2: Average Time to Build Tree:
        plt.figure()
        timesToBuild = [dtTime, rtTime]
        plt.bar(index, timesToBuild)
        plt.xlabel("Type of Tree")
        plt.xticks(index, ["Classic", "Random"])
        plt.legend(loc = 5)
        plt.ylabel("Average Time to Build Tree in Seconds")
        plt.title("Figure 6: \n Average Time to Build Tree: Classic Decision Trees vs. Random Trees")
        plt.savefig("Figure 6.png")


        # Metric 3: Average Time to Query Tree: 
        plt.figure()
        timesToBuild = [dtQTime, rtQTime]
        plt.bar(index, timesToBuild)
        plt.xlabel("Type of Tree")
        plt.xticks(index, ["Classic", "Random"])
        plt.legend(loc = 5)
        plt.ylabel("Average Time to Query Tree in Seconds")
        plt.title("Figure 7: \n Average Time to Query Tree: Classic Decision Trees vs. Random Trees")
        plt.savefig("Figure 7.png")


        if verbose:
            # Metric 1: Average Correlation:
            print
            print "Average In Sample Correlation: "
            print "DTLearner: ", dtINcorr, "   RTLearner: ", rtINcorr
            print
            print "Average Out Sample Correlation: "
            print "DTLearner: ", dtOUTcorr, "   RTLearner: ", rtOUTcorr

            # Metric 2: Average Time to Build Tree:
            print "Average Time to Build Tree: "
            print "DTLearner: ", dtTime
            print
            print "RTLearner: ", rtTime


            # Metric 3: Average Time to Query Tree: 
            print "Average Time to Query Tree: " 
            print "DTLearner: ", dtQTime
            print "RTLearner: ", rtQTime


    def getCorr(self, learner, verbose):
        # evaluate in sample                                                                                
        inpredY = learner.query(self.trainX) # get the predictions                                                                                 
        inCorr = np.corrcoef(inpredY, y=self.trainY)                                                                                

        # evaluate out of sample                                                                                
        outpredY = learner.query(self.testX) # get the predictions 
        outCorr = np.corrcoef(outpredY, y=self.testY)                                                                                 


        if verbose:
                                                                                           
            print "In sample results"                                                                               
            print "corr: ", inCorr[0,1]                                                                              
                                                                                                                                                
            print                                                                               
            print "Out of sample results"                                                                               
            print "corr: ", outCorr[0,1]    
            print


        return inCorr, outCorr


    def getRSME(self, learner, verbose):
        # evaluate in sample                                                                                
        inpredY = learner.query(self.trainX) # get the predictions                                                                                 
        inRMSE = math.sqrt(((self.trainY - inpredY) ** 2).sum()/self.trainY.shape[0])                                                                                 
        
        # evaluate out of sample                                                                                
        outpredY = learner.query(self.testX) # get the predictions                                                                              
        outRSME = math.sqrt(((self.testY - outpredY) ** 2).sum()/self.testY.shape[0])           

        if verbose:
            print                                                                               
            print "In sample results"                                                                               
            print "RMSE: ", inRMSE      
            c = np.corrcoef(inpredY, y=self.trainY)                                                                                
            print "corr: ", c[0,1]                                                                              
                                                                                                                                                
            print                                                                               
            print "Out of sample results"                                                                               
            print "RMSE: ", outRSME                                                                                
            c = np.corrcoef(outpredY, y=self.testY)                                                                                 
            print "corr: ", c[0,1]                  

        return inRMSE, outRSME


    def testALearner(self, learner):
        learner.addEvidence(self.trainX, self.trainY)
        self.getRSME(learner, verbose = True)                                                                              


if __name__=="__main__": 			  		 			     			  	   		   	  			  	
    if len(sys.argv) != 2: 			  		 			     			  	   		   	  			  	
        print "Usage: python testlearner.py <filename>" 			  		 			     			  	   		   	  			  	
        sys.exit(1) 

    
    # Removes first row and column if Istanbul.csv is tested
    with util.get_learner_data_file(sys.argv[1]) as f:
            
            inf = open("Data/" + sys.argv[1]) 
            alldata = np.genfromtxt(f,delimiter=',')                                                                                
            # Skip the date column and header row if we're working on Istanbul data                                                                                 
            if sys.argv[1] == 'Istanbul.csv':                                                                              
                data = alldata[1:,1:]
            else:
                data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()]) 			  		 			     			  	   		   	  			  	
 			  		 	
    
    myTest = tester(data)
    inErr1, outErr1 = myTest.question1()
    myTest.question2(inErr1, outErr1)
    myTest.question3(False)


    # FOR TESTING PURPOSES: Create an instance of all learners to test:

    #learners = []			  		 			     			  	   		   	  			  	
    #learners.append(lrl.LinRegLearner(False)) # create a LinRegLearner 			  		 			     			  	   		   	  			  	
    #learners.append(dt.DTLearner(50, False))
    #learners.append(rt.RTLearner(50, False))
    #learners.append(bl.BagLearner(dt.DTLearner, {"leaf_size" : 20}, 1, False, False))        
    #learners.append(bl.BagLearner(dt.DTLearner, {"leaf_size" : 20}, 20, False, False))

    #learners.append(it.InsaneLearner(False))

    #for learner in learners:
    #    myTest.testALearner(learner)

    #print
    #print "All Learners Tested - You're good to go!"

    
    #print "End of Tester" 
    #print










     			     			  	   		   	  			  	
