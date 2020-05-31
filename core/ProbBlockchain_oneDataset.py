#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:43:11 2018

@author: tarasalman
"""
'''
#This code simulates the probabilstic blockchain concept which combines blockchains and ML
There is no actual network in the middle. Just the basic fucntions including agents predictions using
machine learning algorithms and block generations. 

Machine learning algorithms used here were random forest, 
decision tree, and logistic regression. Others can be used but there were used as they were fast and relatively 
accurate. One dataset was used which is taken from UNSW_NB15 
(https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

The block generation include a PoW mining algorithm (that was commented for faster processing). Block structure 
includes time stamp, agents' predictions, miner ID, and summary of decisions. The summary function used is a list
of mean, mode, and median. 

Client class if for clients 
Miner class if for mining
One run class if for mean running

note to self. These class should be in seperate files before publishing the code
'''
 
import numpy
from scipy import stats
import random
import time
import datetime
import string
from multiprocessing import Queue
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

##initializing variables 
blockResult = multiprocessing.Manager().Queue()
validation_results= Queue()
blocks=[]
miners=[]
svotes=[]
svotes_N=[]
validation_results_N= Queue()
valid_results=[]
validity=0
rand=[]
valid_results=[]
valid_results_N=[]
Results_Prob=[]
RF_Results =[]
DT_Results =[]
LR_Results = []
attacks_detected= []
cm_LR=0
cm_DT=0
cm_PB=0
cm_RF=0

# For machine learning algorithms initializations 
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
def id_generator(size=3, chars=string.ascii_uppercase + string.digits): return ''.join(random.choice(chars) for _ in range(size))
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0)

# Some handon calculating of the the evaluation metrics 
def accuracy(cm):
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp = cm[1,1]
    total_sample = tp+fn+fp+tn
    ac = (tp+tn)/total_sample
    return ac

def FAR(cm):
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp= cm[1,1]
    total_sample = tp+fn+fp+tn
    if fp == 0:
        return 0
    else:
        far = fp/(fp+tn)
        return far  

def UND(cm):
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp = cm[1,1]
    total_sample = tp+fn+fp+tn
    if fn == 0:
        return 0
    else:
        und = fn/(fn+tp)
        return und
    
## A client class initiate clients (or agents) node that will be running either machine learning algorithms or randomly selecting a rand value (malicious) 

class Clients:
    
    def Generate(self, number_of_clients):
        x= []
        for i in range (0, number_of_clients): 
            randi= random.betavariate( 5.67 , 3)
            rand.append(randi)
            x.append([i,randi])
        return(x)
    def Predict(self, number_of_clients, prediction):
        x= []
        global DT_Results
        global LR_Results 
        global RF_Results 
        for i in range (0, number_of_clients): 
            if i ==0 or i==1 or i==2: 
                if i%3==0: 
                    randi =classifier_DT.predict(prediction.reshape(1, -1))
                    if i==0:
                        DT_Results.append(randi[0])
                    x.append([i,randi[0],'DT'])
                elif i%3==1:
                    randi =classifier_RF.predict(prediction.reshape(1, -1))
                    if i==1:
                        RF_Results.append(randi[0])
                    x.append([i,randi[0],'RF'])     
                else:
                    randi =classifier_LR.predict(prediction.reshape(1, -1))
                    if i==2:
                        LR_Results.append(randi[0])
                    x.append([i,randi[0],'LR'])  
            else:
                    c= random.randint(0,2)
                    if c%3==0: 
                        randi =classifier_DT.predict(prediction.reshape(1, -1))
                        x.append([i,randi[0],'DT'])
                    elif c%3==1:
                        randi =classifier_RF.predict(prediction.reshape(1, -1))
                        x.append([i,randi[0],'RF'])     
                    else:
                       randi =classifier_LR.predict(prediction.reshape(1, -1))
                       x.append([i,randi[0],'LR'])  
                    
        return(x)


##Miners contruct blocks 
## Block structure 
        #ratings: predictions made by agents
        #summary [mean (vote cast, meadian, mode, std...)]
        #timestamp for the block
        #miner ID
        #proof of work hash in case we are using it 
        
class miner(): 
    def __init__(self):
        self.svotes= []
        self.minerID=0 
        self.svotes_N=[]
    def run(self):
        print("I am a miner and i am mining")
        #cast= numpy.mean(self.svotes, axis=0)
        #print(votes)
        block={}
        string = id_generator()
        block["ratings"]=self.svotes
        self.svotes= numpy.array(self.svotes)[:,1]
       # print(self.svotes)
        cast= numpy.mean(self.svotes.astype(numpy.float), axis=0)
        mode= stats.mode(self.svotes.astype(numpy.float), axis=0)
        median= numpy.median(self.svotes.astype(numpy.float), axis=0)
        #self.svotes= votes
        #print (self.svotes)
        block["vote_cast"] = cast
        block["mode"] = mode
        block["median"] = median
        cast= numpy.mean(self.svotes_N, axis=0)
        mode2= stats.mode(self.svotes_N, axis=0)
        block["mode2"] = mode2[0][0][1]
        block["vote_cast2"] = cast[1]
        median2= numpy.median(self.svotes_N, axis=0)
        block["median2"] = median2[1]

        cast_std=numpy.std(self.svotes.astype(numpy.float), axis=0)
        block["vote_cast_std"] = cast_std
        cast_std=numpy.std(self.svotes_N, axis=0)
        block["vote_cast_std2"] = cast_std[1]
# =============================================================================
#         while complete == False:
#             curr_string = string + str(n)
#             curr_hash = hashlib.md5(bytes(curr_string,"ascii")).hexdigest()
#             n= n + 1
#             # slows performance drastically
#             #print (curr_hash) 
# 
#             if curr_hash.startswith('0'):
#                 block["proof of work"] = curr_hash
#                 complete = True
# =============================================================================
        ts = time.time()
        #print("I finished mining")
        block["time_stamps"]=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        block["miner_ID"]= self.minerID
       # randomSleep=random.random()
        #sleep(randomSleep)
        blockResult.put(block)
       # print("I am here")
    def set_votes(self,votes):
        
       # print(self.svotes)
        self.svotes=votes
        for i in range (0, len(votes)): 
            #rand= random.random()
            if (votes[i][1] >0.5):
                self.svotes_N.append([i,1])
            else:
                self.svotes_N.append([i,0])
    def set_miner_ID(self,ID):
        self.minerID=ID
    def validate(self, new_block):
        print(" I am validating a block created by",new_block["miner_ID"])
        votes= numpy.mean(numpy.array(self.svotes)[:,1].astype(numpy.float), axis=0)
        cast= votes
        votes= numpy.mean(self.svotes_N, axis=0)
        cast2= votes[1]
        print (new_block["vote_cast"],cast,new_block["vote_cast2"],cast2)
        if  (cast-0.01<=new_block["vote_cast"]<=cast+0.01):
            validation_results.put(1)
        else: 
            validation_results.put(0)
        if (cast2-0.01<=new_block["vote_cast2"]<=cast2+0.01):
            validation_results_N.put(1)
        else: 
            validation_results_N.put(0)
            
# One run controls the miners and clinets 
#Arg number of clients, number of Miners, the testing features and the testing set for checking            
    
def OneRun(numberofClients,numberofMiners, X_test,y_test):
    w=0
    accept_prob=0
    accept_normal=0
    global blocks
    blocks=[]
    global attacks_detected
    attacks_detected =[]
#    blockResult = multiprocessing.Queue()
    while(w < len(X_test)): 
        Gen_Client = Clients()
        global Results_Prob
        Results_Prob= Gen_Client.Predict(numberofClients,X_test[w] )
        processs = []
        miners=[]
        
#        blockResult = multiprocessing.Queue()
        valid_results=[]
        valid_results_N=[]
        validity=0
        while (validity!=1):
            for i in range (numberofMiners):
                mineri =miner()
                if i == 1:
                    mineri.set_votes(Results_Prob)
                elif i == 2:
                    mineri.set_votes(Results_Prob)
                else:
                    mineri.set_votes(Results_Prob)
                mineri.set_miner_ID(i)
                miners.append(mineri)
                process = multiprocessing.Process(target=mineri.run)
                process.start()
                processs.append(process)
            print ("Waiting for result...")
            new_block = blockResult.get(True,2) # waits until any of the proccess have `.put()` a result
            for process in processs: # then kill them all off
                process.terminate()
            #print("I terminated processes")    
            for i in range (numberofMiners):
                process = multiprocessing.Process(target=miners[i].validate, args=[new_block])
                process.start()
                processs.append(process)
                valid_results.append(validation_results.get()) 
                valid_results_N.append(validation_results_N.get()) 
                process.join()
                process.terminate()
           
               
            if (sum(valid_results)>=4):
                blocks.append(new_block)
             #   print("block appended")
                #validity=1
                accept_prob=accept_prob+1 
                if (new_block['vote_cast']>=0.5):
                    attacks_detected.append(1)
                else:
                    attacks_detected.append(0)
            #print(blocks) 
                validity=1
       
        
           
            if (sum(valid_results_N)>=4):
                # blocks.append(new_block)
             #   print("block appended")
                #  validity=1
                accept_normal=accept_normal+1 
                #print(blocks) 
        w=w+1
        print(w)
    global cm_DT
    global cm_RF
    global cm_PB
    global cm_LR
    print ("normal accept",accept_normal)
    print ("probablisitic accept",accept_prob," \n",w )
    cm_DT = confusion_matrix(y_test, DT_Results)
    print (cm_DT)
    arq.writelines("%s"%cm_DT)
    cm_RF = confusion_matrix(y_test, RF_Results)
    print (cm_RF)
    arq.writelines("%s"%cm_RF)
    cm_PB = confusion_matrix(y_test, attacks_detected)
    print (cm_PB)
    arq.writelines("%s"%cm_PB)
    cm_LR = confusion_matrix(y_test, LR_Results)
    print (cm_LR)
    arq.writelines("%s"%cm_LR)


def RandomForestTrain(X_train, y_train): 
    #-----------------Random Forest----------------------------------------
    print("Inicializing Random Forest\n")
    # Fitting Random Forest Classification to the Training set
    classifier_RF.fit(X_train, y_train)
    # Predicting the Test set results
    #y_pred_RF = classifier_RF.predict(X_test)
    # save the model to disk
    #filename = 'finalized_RF_model.sav'
    #pickle.dump(classifier_RF, open(filename, 'wb'))
    
def Logistic_RegressionTrain(X_train, y_train): 
    #-----------------Logistic Regression----------------------------------------
    print("Inicializing Logistic Regresssion \n")
    # Fitting Random Forest Classification to the Training set
    classifier_LR.fit(X_train, y_train)
    # Predicting the Test set results
    #y_pred_RF = classifier_RF.predict(X_test)
    # save the model to disk
    #filename = 'finalized_RF_model.sav'
    #pickle.dump(classifier_RF, open(filename, 'wb'))
def DecisionTreeTrain(X_train, y_train): 
    #-----------------Decision Tree----------------------------------------
    print("Inicializing Decision Tree \n")
    # Fitting Random Forest Classification to the Training set
    classifier_DT.fit(X_train, y_train)
    # Predicting the Test set results
    #y_pred_RF = classifier_RF.predict(X_test)
    # save the model to disk
    #filename = 'finalized_RF_model.sav'
    #pickle.dump(classifier_RF, open(filename, 'wb'))   
    
###Training Machine learning models     

# Importing the dataset
dataset_tain = pd.read_csv('Datasets/dos-training.csv', sep=",")

#dataset = pd.read_csv('/root/Downloads/traffic-v2.csv', sep=";")
X_train = dataset_tain.iloc[:,0:38].values
y_train = dataset_tain.iloc[:,39].values

#print(X_train)
#print(y_train)
dataset_test = pd.read_csv('Datasets/DoS-testing.csv', sep=",")
X_test = dataset_test.iloc[:, 0:38].values
y_test = dataset_test.iloc[:,39].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.10, random_state = 0)
RandomForestTrain(X_train, y_train)
DecisionTreeTrain(X_train, y_train)
Logistic_RegressionTrain(X_train, y_train)
arq = open('Test10.csv', 'w')  
xx= [10]
for i in range (0,len(xx)):
    OneRun(xx[i],5, X_test,y_test)
    svotes=[]
    svotes_N=[]
   # rand=[]
    print ("%d iteration"%i)
arq.close()


Algorithms = ('RF_Model', 'DT_Model', 'LR_Model','PB')
y_pos = numpy.arange(len(Algorithms))

ac = [accuracy(cm_RF)*100, accuracy(cm_DT)*100, accuracy(cm_LR)*100, accuracy(cm_PB)*100]

far = [FAR(cm_RF)*100, FAR(cm_DT)*100, FAR(cm_LR)*100, FAR(cm_PB)*100]

und = [UND(cm_RF)*100, UND(cm_DT)*100, UND(cm_LR)*100,UND(cm_PB)*100]

fig = plt.figure()
plt.bar(y_pos, ac, align='center', alpha=0.5)
for a,b in zip(y_pos, ac):
    plt.text(a, b, str(round(b,2)))
plt.xticks(y_pos, Algorithms)
plt.ylabel('Accuracy')
plt.title('Algorithms')
plt.show()
fig.savefig('Figures/ACC_Case1.jpeg')

fig = plt.figure()
plt.bar(y_pos, far, align='center', alpha=0.5)
for a,b in zip(y_pos, far):
    plt.text(a, b, str(round(b,2)))
plt.xticks(y_pos, Algorithms)
plt.ylabel('False Alarme Rate')
plt.title('Algorithms')
plt.show()
fig.savefig('Figures/FAR_Case1.jpeg')

fig = plt.figure()
plt.bar(y_pos, und, align='center', alpha=0.5)
for a,b in zip(y_pos,und):
    plt.text(a, b, str(round(b,2)))
plt.xticks(y_pos, Algorithms)
plt.ylabel('Un-Detection Rate')
plt.title('Algorithms')
plt.show()
fig.savefig('Figures/UND_Case1.jpeg')
