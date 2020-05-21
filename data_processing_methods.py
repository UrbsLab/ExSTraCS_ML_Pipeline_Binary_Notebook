# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:26:40 2019

@author: Ryan Urbanowicz - University of Pennsylvania
Includes methods to perform specialized partitioning of a dataset for cross 
validation
"""
import pandas as pd
import numpy as np
import pickle

#Import data transformation packages
from sklearn.preprocessing import StandardScaler

#Import data imputation packages
#from fancyimpute import IterativeImputer # a.k.a MICE - Recommended advanced method / Industry standard
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer # a.k.a MICE - Recommended advanced method / Industry standard

def cv_partitioner(td, cv_partitions, partition_method, outcomeLabel, categoricalOutcome, matchName, randomSeed):
    """ Takes data frame (td), number of cv partitions, partition method 
    (R, S, or M), outcome label, Boolean indicated whether outcome is categorical
    and the column name used for matched CV. Returns list of training and testing
    dataframe partitions.
    """
    #Partitioning-----------------------------------------------------------------------------------------
    #Shuffle instances to avoid potential biases
    td = td.sample(frac=1, random_state = randomSeed).reset_index(drop=True)
                
    #Temporarily convert data frame to list of lists (save header for later)
    header = list(td.columns.values)
    datasetList = list(list(x) for x in zip(*(td[x].values.tolist() for x in td.columns)))

    #Handle Special Variables for Nominal Outcomes
    outcomeIndex = None
    classList = None
    if categoricalOutcome:
        outcomeIndex = td.columns.get_loc(outcomeLabel)
        classList = []
        for each in datasetList:
            if each[outcomeIndex] not in classList:
                classList.append(each[outcomeIndex])
                
    #Initialize partitions
    partList = [] #Will store partitions
    for x in range(cv_partitions):
        partList.append([])
    
    #Random Partitioning Method----------------------------
    if partition_method == 'R':
        print("Random Partitioning")
        currPart = 0
        counter = 0
        for row in datasetList:
            partList[currPart].append(row)
            counter += 1
            currPart = counter%cv_partitions
    
    #Stratified Partitioning Method-----------------------
    elif partition_method == 'S':
        if categoricalOutcome: #Discrete outcome
            print("Nominal Stratitifed Partitioning")
            
            #Create data sublists, each having all rows with the same class
            byClassRows = [ [] for i in range(len(classList)) ] #create list of empty lists (one for each class)
            for row in datasetList:
                #find index in classList corresponding to the class of the current row. 
                cIndex = classList.index(row[outcomeIndex])
                byClassRows[cIndex].append(row)

            for classSet in byClassRows:
                currPart = 0
                counter = 0
                for row in classSet:
                    partList[currPart].append(row)
                    counter += 1
                    currPart = counter%cv_partitions
    
        else: # Do stratified partitioning for continuous endpoint data
            print("Error: Stratified partitioning only designed for discrete endpoints. ")
    
    elif partition_method == 'M':
        if categoricalOutcome:
            #Get match variable column index
            outcomeIndex = td.columns.get_loc(outcomeLabel)
            matchIndex = td.columns.get_loc(matchName)

            print("Nominal Matched Partitioning")
            #Create data sublists, each having all rows with the same match identifier
            matchList = []
            for each in datasetList:
                if each[matchIndex] not in matchList:
                    matchList.append(each[matchIndex])

            byMatchRows = [ [] for i in range(len(matchList)) ] #create list of empty lists (one for each match group)
            for row in datasetList:
                #find index in matchList corresponding to the matchset of the current row. 
                mIndex = matchList.index(row[matchIndex])
                row.pop(matchIndex) #remove match column from partition output
                byMatchRows[mIndex].append(row)
                    
            currPart = 0
            counter = 0
            for matchSet in byMatchRows: #Go through each unique set of matched instances
                for row in matchSet: #put all of the instances
                    partList[currPart].append(row)
                #move on to next matchset being placed in the next partition. 
                counter += 1
                currPart = counter%cv_partitions

            header.pop(matchIndex) #remove match column from partition output
        else: 
            print("Error: Matched partitioning only designed for discrete endpoints. ")
            
    else:
        print('Error: Requested partition method not found.')
        
   #Generation of CV datasets from partitions---------------------------------------------------------------------------
    train_dfs = []
    test_dfs = []
    for part in range(0, cv_partitions):
        testList=partList[part] # Assign testing set as the current partition

        trainList=[]
        tempList = []                 
        for x in range(0,cv_partitions): 
            tempList.append(x)                            
        tempList.pop(part)

        for v in tempList: #for each training partition
            trainList.extend(partList[v])   
  
        train_dfs.append(pd.DataFrame(trainList, columns = header))
        test_dfs.append(pd.DataFrame(testList, columns = header))
            
    return train_dfs, test_dfs 


def identifyCategoricalFeatures(x_data,categoricalCutoff):
    """ Takes a dataframe (of independent variables) with column labels and returns a list of column names identified as 
    being categorical based on user defined cutoff. """
    categorical_variables = []
    for each in x_data:
        if x_data[each].nunique() <= categoricalCutoff:
            categorical_variables.append(each)
    return categorical_variables


def dataScaling(train_dfs, test_dfs, outcomeLabel, instLabel, name_path, header):
    scale_train_dfs = []
    scale_test_dfs = []
    #Scale all training datasets
    i = 0
    for each in train_dfs:
        df = each
        if instLabel == None or instLabel == 'None':
            x_train = df.drop([outcomeLabel], axis=1)
        else:
            x_train = df.drop([outcomeLabel,instLabel], axis=1)
            inst_train = df[instLabel] #pull out instance labels in case they include text
        y_train = df[outcomeLabel]

        #Scale features (x)
        scaler = StandardScaler()
        scaler.fit(x_train) 
        x_train_scaled = pd.DataFrame(scaler.transform(x_train),columns = x_train.columns)

        #Save scalar
        pickle.dump(scaler, open(name_path+str(i)+'.sav', 'wb'))
    
        #Recombine x and y
        if instLabel == None or instLabel == 'None':
            scale_train_dfs.append(pd.concat([pd.DataFrame(y_train, columns = [outcomeLabel]), pd.DataFrame(x_train_scaled, columns = header)], axis=1, sort=False))
        else:
            scale_train_dfs.append(pd.concat([pd.DataFrame(y_train, columns = [outcomeLabel]), pd.DataFrame(inst_train, columns = [instLabel]), pd.DataFrame(x_train_scaled, columns = header)], axis=1, sort=False))

        #Scale corresponding testing dataset
        df = test_dfs[i]
        if instLabel == None or instLabel == 'None':
            x_test = df.drop([outcomeLabel], axis=1)
        else:
            x_test = df.drop([outcomeLabel,instLabel], axis=1)
            inst_test = df[instLabel] #pull out instance labels in case they include text
        y_test = df[outcomeLabel]

        #Scale features (x)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test),columns = x_test.columns)
    
        #Recombine x and y
        if instLabel == None or instLabel == 'None':
            scale_test_dfs.append(pd.concat([pd.DataFrame(y_test, columns = [outcomeLabel]), pd.DataFrame(x_test_scaled, columns = header)], axis=1, sort=False))
        else:
            scale_test_dfs.append(pd.concat([pd.DataFrame(y_test, columns = [outcomeLabel]), pd.DataFrame(inst_test, columns = [instLabel]), pd.DataFrame(x_test_scaled, columns = header)], axis=1, sort=False))
            
        i += 1

    return scale_train_dfs, scale_test_dfs


def imputeCVData(outcomeLabel, instLabel, categorical_variables, header, train_dfs, test_dfs, randomSeed):
    #Begin by imputing categorical variables with simple 'mode' imputation
    imp_train_dfs = []
    imp_test_dfs = []

    #Impute all training datasets
    for each in train_dfs:
        for c in each.columns:
            if c in categorical_variables:
                each[c].fillna(each[c].mode().iloc[0], inplace=True)
        
    #Impute all testing datasets
    for each in test_dfs:
        for c in each.columns:
            if c in categorical_variables:
                each[c].fillna(each[c].mode().iloc[0], inplace=True)
                
    #Now impute remaining ordinal variables
    
    #Impute all training datasets
    for each in train_dfs:
        df = each
        if instLabel == None or instLabel == 'None':
            x_train = df.drop([outcomeLabel], axis=1).values
        else:
            x_train = df.drop([outcomeLabel,instLabel], axis=1).values
            inst_train = df[instLabel].values #pull out instance labels in case they include text
        y_train = df[outcomeLabel].values

        #Impute features (x)
        x_new_train = IterativeImputer(random_state = randomSeed).fit_transform(x_train)

        #Recombine x and y
        if instLabel == None or instLabel == 'None':
            imp_train_dfs.append(pd.concat([pd.DataFrame(y_train, columns = [outcomeLabel]), pd.DataFrame(x_new_train, columns = header)], axis=1, sort=False))
        else:
            imp_train_dfs.append(pd.concat([pd.DataFrame(y_train, columns = [outcomeLabel]), pd.DataFrame(inst_train, columns = [instLabel]), pd.DataFrame(x_new_train, columns = header)], axis=1, sort=False))

    #Impute all testing datasets
    for each in test_dfs:
        df = each
        if instLabel == None or instLabel == 'None':
            x_test = df.drop([outcomeLabel], axis=1).values
        else:
            x_test = df.drop([outcomeLabel,instLabel], axis=1).values
            inst_test = df[instLabel].values #pull out instance labels in case they include text
            
        y_test = df[outcomeLabel].values


        #Impute features (x)
        x_new_test = IterativeImputer(random_state = randomSeed).fit_transform(x_test)

        #Recombine x and y
        if instLabel == None or instLabel == 'None':
            imp_test_dfs.append(pd.concat([pd.DataFrame(y_test, columns = [outcomeLabel]), pd.DataFrame(x_new_test, columns = header)], axis=1, sort=False))
        else:
            imp_test_dfs.append(pd.concat([pd.DataFrame(y_test, columns = [outcomeLabel]),pd.DataFrame(inst_test, columns = [instLabel]), pd.DataFrame(x_new_test, columns = header)], axis=1, sort=False))
    
    return imp_train_dfs, imp_test_dfs
