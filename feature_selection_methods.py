# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:26:40 2019

@author: Ryan Urbanowicz - University of Pennsylvania
Includes methods to facilitate feature selection analysis and summary
"""
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif
from skrebate import MultiSURF

import matplotlib.pyplot as plt


def reportAllFS(scoreSet, algorithm,ordered_feature_names,output_folder,data_name):
    
    df = pd.DataFrame(scoreSet, columns=ordered_feature_names)

    filepath = output_folder+'/'+algorithm+'_FI_'+data_name+'.csv'
    df.to_csv(filepath, index=False)  


def reportTopFS(scoreSum, algorithm, cv_partitions,topResults,wd_path,output_folder,data_name):
    #Make the sum of scores an average
    for v in scoreSum:
        scoreSum[v] = scoreSum[v]/float(cv_partitions)

    #Sort averages (decreasing order and print top 'n' and plot top 'n'
    f_names = []
    f_scores = []
    for each in scoreSum:
        f_names.append(each)
        f_scores.append(scoreSum[each])

    names_scores = {'Names':f_names, 'Scores':f_scores} 
    ns = pd.DataFrame(names_scores)
    ns = ns.sort_values(by='Scores',ascending = False)
    
    #Select top 'n' to report and plot
    ns = ns.head(topResults)

    #Visualize sorted feature scores
    ns['Scores'].plot(kind='barh',figsize=(6,12))
    plt.ylabel('Features')
    plt.xlabel(str(algorithm)+' Score')
    #plt.yticks(np.arange(len(f_names)), ns['Feature Names'])
    plt.yticks(np.arange(len(ns['Names'])), ns['Names'])
    plt.title('Sorted '+str(algorithm)+' Scores')
    plt.savefig((wd_path+output_folder+'/'+algorithm+'_FI_BarPlot_' + data_name), bbox_inches = "tight")


def sort_save_fi_scores(scores, ordered_feature_names, algorithm, filename):
    #Put list of scores in dictionary
    scoreDict = {}
    i=0
    for each in ordered_feature_names:
        scoreDict[each] = scores[i]
        i += 1

    #Sort features by decreasing score
    score_sorted_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
    
    #Save scores to 'formatted' file
    fh = open(filename, 'w')
    fh.write(algorithm +' analysis \n')
    fh.write('Run Time (sec): ' + str('NA') + '\n')
    fh.write('=== SCORES ===\n')
    n = 1
    for k in score_sorted_features:
        fh.write(str(k) + '\t' + str(scoreDict[k])  + '\t' + str(n) +'\n')
        n+=1
    fh.close()
        
    return scoreDict, score_sorted_features


def sort_save_fs_fi_scores(scoreDict, algorithm, filename):
    #Sort features by decreasing score
    score_sorted_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)
	
    #Save scores to 'formatted' file
    fh = open(filename, 'w')
    fh.write(algorithm +' analysis \n')
    fh.write('Run Time (sec): ' + str('NA') + '\n')
    fh.write('=== SCORES ===\n')
    n = 1
    for k in score_sorted_features:
        fh.write(str(k) + '\t' + str(scoreDict[k])  + '\t' + str(n) +'\n')
        n+=1
    fh.close()
	
	
def run_mi(xTrain, yTrain, cv_count, data_name, output_folder, randSeed, ordered_feature_names, algorithm):
    #Run mutual information
    filename = output_folder+'/'+algorithm+'_'+data_name+'_'+str(cv_count)+'_Train.txt'
    scores = mutual_info_classif(xTrain, yTrain, random_state=randSeed)

    scoreDict, score_sorted_features = sort_save_fi_scores(scores, ordered_feature_names, algorithm, filename)
        
    return scores, scoreDict, score_sorted_features


def run_multisurf(xTrain, yTrain, cv_count, data_name, output_folder, randSeed, ordered_feature_names, algorithm):
    #Run mutlisurf
    filename = output_folder+'/'+algorithm+'_'+data_name+'_'+str(cv_count)+'_Train.txt'
    
    clf = MultiSURF().fit(xTrain, yTrain)
    scores = clf.feature_importances_

    scoreDict, score_sorted_features = sort_save_fi_scores(scores, ordered_feature_names, algorithm, filename)
        
    return scores, scoreDict, score_sorted_features


def selectFeatures(algorithms, cv_partitions, selectedFeatureLists, maxFeaturesToKeep,metaFeatureRanks):
    cv_Selected_List = [] #list of selected features for each cv
    numAlgorithms = len(algorithms)
    if numAlgorithms > 1: #'Interesting' features determined by union of feature selection results (from different algorithms)
        for i in range(cv_partitions):
            unionList = selectedFeatureLists[algorithms[0]][i] #grab first algorithm's lists
            #Determine union
            for j in range(1,numAlgorithms): #number of union comparisons
                unionList = list(set(unionList) | set(selectedFeatureLists[algorithms[j]][i]))

            if len(unionList) > maxFeaturesToKeep: #Apply further filtering if more than max features remains
                #Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    for each in metaFeatureRanks:
                        targetFeature = metaFeatureRanks[each][i][k]
                        if not targetFeature in newFeatureList:
                            newFeatureList.append(targetFeature)
                        if len(newFeatureList) < maxFeaturesToKeep:
                            break
                    k += 1
                unionList = newFeatureList
            unionList.sort()  #Added to ensure script random seed reproducibility
            cv_Selected_List.append(unionList)

    else: #Only one algorithm applied
        for i in range(cv_partitions):
            featureList = selectedFeatureLists[algorithms[0]][i] #grab first algorithm's lists

            if len(featureList) > maxFeaturesToKeep: #Apply further filtering if more than max features remains
                #Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    targetFeature = metaFeatureRanks[algorithms[0]][i][k]
                    newFeatureList.append(targetFeature)
                    k+=1
            cv_Selected_List.append(newFeatureList)
            
    return cv_Selected_List


def genFilteredDatasets(cv_Selected_List, outcomeLabel, instLabel,cv_partitions,cv_data_folder,data_name):
    #create lists to hold training and testing set dataframes.
    trainList = []
    testList = []

    for i in range(cv_partitions):
        #Load training partition
        trainSet = pd.read_csv(cv_data_folder+'/'+data_name+'_'+str(i)+'_Train.txt', na_values='NA', sep = "\t")
        trainList.append(trainSet)

        #Load testing partition
        testSet = pd.read_csv(cv_data_folder+'/'+data_name+'_'+str(i)+'_Test.txt', na_values='NA', sep = "\t")
        testList.append(testSet)

        #Training datasets
        labelList = [outcomeLabel]
        if not instLabel == None:
            labelList.append(instLabel)
        labelList = labelList + cv_Selected_List[i]

        td_train = trainList[i][labelList]
        td_train.to_csv(cv_data_folder+'/'+data_name+'_FS_'+str(i)+'_Train.txt', index=None, sep='\t')

        td_test = testList[i][labelList]
        td_test.to_csv(cv_data_folder+'/'+data_name+'_FS_'+str(i)+'_Test.txt', index=None, sep='\t')
