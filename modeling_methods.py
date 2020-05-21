# -*- coding: utf-8 -*-
"""
Created on Sat Dec  16 14:26:40 2019

@author: Ryan Urbanowicz - University of Pennsylvania
Includes methods to facilitate machine learning modeling and summary
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pickle

#Scikit-Learn Packages:
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
#import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics

import xgboost as xgb
import lightgbm as lgb
import optuna #hyperparameter optimization
import plotly
from scipy import interp

#Import Progress bar:
from tqdm import tnrange, tqdm_notebook


def classEval(y_true, y_pred, verbose = False):
	#calculate and store evaluation metrics
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	ac = accuracy_score(y_true, y_pred)
	bac = balanced_accuracy_score(y_true, y_pred)
	re = recall_score(y_true, y_pred)
	pr = precision_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)

	#calculate specificity
	if tn == 0 and fp == 0:
		sp = 0
	else:
		sp = tn/float(tn+fp)

	if verbose:
		
		print("Balanced Accuracy:    "+str(bac))
		print("Accuracy (Standard):  "+str(ac))
		print("F1 Score:             "+str(f1))
		print("Sensitivity (Recall): "+str(re))
		print("Specificity:          "+str(sp))
		print("Precision :           "+str(pr))
		print("TP:                   "+str(tp))
		print("TN:                   "+str(tn))
		print("FP:                   "+str(fp))
		print("FN:                   "+str(fn))

	return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn]


def roc_plot_single(fpr, tpr, roc_auc):
        #Plot the ROC Curve and include AUC in figure. 
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    
def save_performance(algorithm,s_bac,s_ac,s_f1,s_re,s_sp,s_pr,s_tp,s_tn,s_fp,s_fn,aucs,praucs,aveprecs,output_folder,data_name):
    
        results = {'Balanced Accuracy':s_bac, 'Accuracy':s_ac, 'F1_Score':s_f1,
                'Recall':s_re, 'Specificity':s_sp, 'Precision':s_pr, 'TP':s_tp, 'TN':s_tn, 'FP':s_fp, 'FN':s_fn, 'ROC_AUC':aucs, 'PRC_AUC':praucs, 'PRC_APS':aveprecs}

        dr = pd.DataFrame(results)
 
        filepath = output_folder+'/'+algorithm+'_Metrics_'+data_name+'.csv'
        dr.to_csv(filepath, header=True, index=False)  
    
    
def save_FI(FI_all, algorithm,data_name,globalFeatureList,output_folder):
        dr = pd.DataFrame(FI_all)
        filepath = output_folder+'/'+algorithm+'_FI_'+data_name+'.csv'
        dr.to_csv(filepath, header=globalFeatureList, index=False)  


def eval_Algorithm_FI(algorithm,ordered_feature_names,xTrainList,yTrainList,xTestList,yTestList,cv_partitions,global_ordered_features,wd_path,output_folder,data_name,randSeed,param_grid,model_folder,algColor,hype_cv,n_trials,scoring_metric,timeout):
	alg_result_table = []
	#Define evaluation stats variable lists
	s_bac = []
	s_ac = []
	s_f1 = []
	s_re = []
	s_sp = []
	s_pr = []
	s_tp = []
	s_tn = []
	s_fp = []
	s_fn = []

	#Define feature importance lists
	FI_all = []
	FI_ave = [0]*len(ordered_feature_names) #Holds only the selected feature FI results for each partition

	#Define ROC plot variable lists
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)
	#Define PRC plot variable lists
	precs = []
	praucs = []
	aveprecs = []

	#Pickle model name basis
	name_path = wd_path+model_folder+'/'+'Model_' +algorithm+'_'+ data_name+'_'

	for i in tqdm_notebook(range(cv_partitions), desc='1st loop'):
		#Algorithm Specific Code
		print("Running "+str(algorithm))
		if algorithm == 'logistic_regression':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_LR_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'decision_tree':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_DT_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'random_forest':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_RF_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'naive_bayes':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_NB_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],i,name_path)
		elif algorithm == 'XGB':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_XGB_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'LGB':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_LGB_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'ANN':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_ANN_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		elif algorithm == 'SVM':
			metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi = run_SVM_full(xTrainList[i], yTrainList[i], xTestList[i], yTestList[i],randSeed,i,param_grid[algorithm],name_path,hype_cv,n_trials,scoring_metric,timeout,wd_path,output_folder,algorithm,data_name)
		else:
			print("Error: Algorithm not found!")

		#Update evaluation stats variable lists [bac, ac, f1, re, sp, pr, tp, tn, fp, fn]
		s_bac.append(metricList[0])
		s_ac.append(metricList[1])
		s_f1.append(metricList[2])
		s_re.append(metricList[3])
		s_sp.append(metricList[4])
		s_pr.append(metricList[5])
		s_tp.append(metricList[6])
		s_tn.append(metricList[7])
		s_fp.append(metricList[8])
		s_fn.append(metricList[9])

		alg_result_table.append([fpr,tpr,roc_auc,recall,prec,prec_rec_auc,ave_prec])

		#Update ROC plot variable lists
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		aucs.append(roc_auc)

		#Update PRC plot variable lists
		precs.append(interp(mean_recall, recall, prec))
		praucs.append(prec_rec_auc)
		aveprecs.append(ave_prec)
        
		#Format feature importance scores as list (takes into account that all features are not in each CV partition)
		tempList = []
		j = 0
		for each in ordered_feature_names:
			if each in global_ordered_features[i]: #Check if current feature from original dataset was in the partition
				#Deal with features not being in original order (find index of current feature list.index()
				f_index = global_ordered_features[i].index(each)
				FI_ave[j] += fi[f_index]
				tempList.append(fi[f_index])
			else:
				tempList.append(0)
			j += 1
			
		FI_all.append(tempList)

	#ROC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
	plt.figure(figsize=(7,7))
	for i in range(cv_partitions):
		#Plot individual CV ROC line
		plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3,
				 label='ROC fold %d (AUC = %0.2f)' % (i, alg_result_table[i][2]))
		
	#Add chance line to ROC plot
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
			 label='Chance', alpha=.8)

	#Calculate and draw average CV ROC line     list.index(element)
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color=algColor,
			 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
			 lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(algorithm+' : ROC over CV Partitions')
	plt.legend(loc="best")
	plt.savefig((wd_path+output_folder+'/'+algorithm+'_ROC_' + data_name), bbox_inches = "tight")
	plt.show()

	#PRC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
		#https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/
		#https://amirhessam88.github.io/roc-vs-pr/
		#https://stackoverflow.com/questions/55541254/precision-recall-curve-with-n-fold-cross-validation-showing-standard-deviation
	plt.figure(figsize=(7,7))
	for i in range(cv_partitions):
		#Plot individual CV ROC line
		plt.plot(alg_result_table[i][3], alg_result_table[i][4], lw=1, alpha=0.3,
				 label='PRC fold %d (AUC = %0.2f) (APS = %0.2f)' % (i, alg_result_table[i][5],alg_result_table[i][6]))
		
	#Add chance line to PRC plot
	noskill = len(yTestList[0][yTestList[0]==1]) / len(yTestList[0]) #Fraction of cases
	plt.plot([0, 1], [noskill, noskill], linestyle='--', lw=2, color='r',
			 label='Chance', alpha=.8)

	#Calculate and draw average CV PRC line
	mean_prec = np.mean(precs, axis=0)
	mean_pr_auc = auc(mean_recall, mean_prec)
	std_auc = np.std(praucs)
	plt.plot(mean_recall, mean_prec, color=algColor,
			 label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_pr_auc, std_auc),
			 lw=2, alpha=.8)

	std_prec = np.std(precs, axis=0)
	precs_upper = np.minimum(mean_prec + std_prec, 1)
	precs_lower = np.maximum(mean_prec - std_prec, 0)
	plt.fill_between(mean_fpr, precs_lower, precs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(algorithm+' : PRC over CV Partitions')
	plt.legend(loc="best")
	plt.savefig((wd_path+output_folder+'/'+algorithm+'_PRC_' + data_name), bbox_inches = "tight")
	plt.show()

	#Calculate and report average eval statistics.
	print("Avg. Model Balanced Accuracy = " + str(np.mean(s_bac)) +
		 " (std. dev. = " + str(np.std(s_bac)) + ")")    
	print("Avg. Model Accuracy = " + str(np.mean(s_ac)) +
		  " (std. dev. = " + str(np.std(s_ac)) + ")")
	print("Avg. Model F1-Score = " + str(np.mean(s_f1)) +
		  " (std. dev. = " + str(np.std(s_f1)) + ")")
	print("Avg. Model Recall = " + str(np.mean(s_re)) +
		 " (std. dev. = " + str(np.std(s_re)) + ")")
	print("Avg. Model Specificity = " + str(np.mean(s_sp)) +
		 " (std. dev. = " + str(np.std(s_sp)) + ")")
	print("Avg. Model Precision = " + str(np.mean(s_pr)) +
		  " (std. dev. = " + str(np.std(s_pr)) + ")")
	print("Avg. Model True Positives = " + str(np.mean(s_tp)) +
		  " (std. dev. = " + str(np.std(s_tp)) + ")")
	print("Avg. Model True Negatives = " + str(np.mean(s_tn)) +
		  " (std. dev. = " + str(np.std(s_tn)) + ")")
	print("Avg. Model False Positives = " + str(np.mean(s_fp)) +
		  " (std. dev. = " + str(np.std(s_fp)) + ")")
	print("Avg. Model False Negatives = " + str(np.mean(s_fn)) +
		  " (std. dev. = " + str(np.std(s_fn)) + ")")
	print("Avg. ROC AUC = " + str(np.mean(aucs)) +
		  " (std. dev. = " + str(np.std(aucs)) + ")")
	print("Avg. PRC AUC = " + str(np.mean(praucs)) +
		  " (std. dev. = " + str(np.std(praucs)) + ")")
	print("Avg. PRC Precision Score = " + str(np.mean(aveprecs)) +
		  " (std. dev. = " + str(np.std(aveprecs)) + ")")

	# Save metrics for printing to file. 
	save_performance(algorithm,s_bac,s_ac,s_f1,s_re,s_sp,s_pr,s_tp,s_tn,s_fp,s_fn,aucs,praucs,aveprecs,output_folder,data_name)

	#Calculate average feature importance scores
	for i in range(0,len(FI_ave)):
		FI_ave[i] = FI_ave[i]/float(cv_partitions)

	#save feature importance scores to file
	save_FI(FI_all, algorithm,data_name,ordered_feature_names,output_folder)
	mean_ave_prec = np.mean(aveprecs)
	return mean_fpr, mean_tpr, mean_auc, mean_prec, mean_pr_auc, mean_ave_prec, FI_ave


def computeImportances(clf, x_train, y_train, x_test, y_test, bac):
        #Reruns the model n times (once for each feature), and evaluates performance change as a standard of feature importance

        feature_count = len(x_train[0])
        #print(feature_count)
        FIbAccList = []
        for feature in tqdm_notebook(range(feature_count), desc='1st loop'):
                indexList = []
                indexList.extend(range(0, feature))
                indexList.extend(range(feature + 1, feature_count))
        
                #Create temporary training and testing sets
                tempTrain = pd.DataFrame(x_train)
                FIxTrainList = tempTrain.iloc[:, indexList].values

                tempTest = pd.DataFrame(x_test)
                FIxTestList = tempTest.iloc[:, indexList].values
        
                clf.fit(FIxTrainList, y_train)
                FIyPred = clf.predict(FIxTestList)
        
                FIbAccList.append(balanced_accuracy_score(y_test, FIyPred))
    
        #Lower balanced accuracy metric values suggest higher feature importance
        featureImpList = []
        for element in FIbAccList:
                if element > bac: #removal of feature yielded higher accuracy
                        featureImpList.append(0) #worst importance
                else:
                        featureImpList.append(bac - element)
        
        return featureImpList

def intersection(lst1, lst2): 
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3 
	
	
def hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric):
	cv = StratifiedKFold(n_splits=hype_cv, shuffle=True, random_state=randSeed)          
	model = clone(est).set_params(**params)  
	#Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
	for a in ['random_state','seed']:
		if hasattr(model,a):
			setattr(model,a,randSeed)
	performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric )) 
	return performance
	
	
def objective_LR(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
	params = {'penalty' : trial.suggest_categorical('penalty',param_grid['penalty']),
                'dual' : trial.suggest_categorical('dual', param_grid['dual']),
                'C' : trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
                'solver' : trial.suggest_categorical('solver',param_grid['solver']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
                'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1]),
                'n_jobs' : trial.suggest_categorical('n_jobs',param_grid['n_jobs'])}
	
	return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

	
def run_LR_full(x_train, y_train, x_test, y_test, randSeed, i, param_grid, name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est = LogisticRegression()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_LR(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = LogisticRegression()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and AUC
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	#fi = np.exp(clf.coef_[0]) Estimate from coeficients (potentially unreliable even with data scaling)
	fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

	
def objective_DT(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
	params = {'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'splitter' : trial.suggest_categorical('splitter', param_grid['splitter']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight'])}
	
	return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)
	
	
def run_DT_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est =tree.DecisionTreeClassifier()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_DT(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = tree.DecisionTreeClassifier()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	fi = clf.feature_importances_

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

def objective_RF(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
	params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'bootstrap' : trial.suggest_categorical('bootstrap',param_grid['bootstrap']),
                'oob_score' : trial.suggest_categorical('oob_score',param_grid['oob_score']),
                'n_jobs' : trial.suggest_categorical('n_jobs',param_grid['n_jobs']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight'])}
	
	return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)


def run_RF_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est = RandomForestClassifier()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_RF(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = RandomForestClassifier()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	fi = clf.feature_importances_

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi


def run_NB_full(x_train, y_train, x_test, y_test,i,name_path):
        #No hyperparameters to optimize.

        #Train model using 'best' hyperparameters - Uses default 3-fold internal CV (training/validation splits)
        clf = GaussianNB()

        model = clf.fit(x_train, y_train)

        #Save model
        pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))
    
        #Prediction evaluation
        y_pred = clf.predict(x_test)

        metricList = classEval(y_test, y_pred, False)

        #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
        probas_ = model.predict_proba(x_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
    
        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])
    
        #Feature Importance Estimates
        fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])
    
        return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

	
def objective_XGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
        posInst = sum(y_train)
        negInst = len(y_train) - posInst
        classWeight = negInst/float(posInst)
        
        params = {'booster' : trial.suggest_categorical('booster',param_grid['booster']),
                'objective' : trial.suggest_categorical('objective',param_grid['objective']),
                'verbosity' : trial.suggest_categorical('verbosity',param_grid['verbosity']),
                'reg_lambda' : trial.suggest_loguniform('reg_lambda', param_grid['reg_lambda'][0], param_grid['reg_lambda'][1]),
                'alpha' : trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),	
                'eta' : trial.suggest_loguniform('eta', param_grid['eta'][0], param_grid['eta'][1]),			
                'gamma' : trial.suggest_loguniform('gamma', param_grid['gamma'][0], param_grid['gamma'][1]),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'grow_policy' : trial.suggest_categorical('grow_policy',param_grid['grow_policy']),
                'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'subsample' : trial.suggest_uniform('subsample', param_grid['subsample'][0], param_grid['subsample'][1]),
                'min_child_weight' : trial.suggest_loguniform('min_child_weight', param_grid['min_child_weight'][0], param_grid['min_child_weight'][1]),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
                'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1.0, classWeight])}
        
        return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)
	

def run_XGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est = xgb.XGBClassifier()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_XGB(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = xgb.XGBClassifier()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi


def objective_LGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
        posInst = sum(y_train)
        negInst = len(y_train)-posInst
        classWeight = negInst/float(posInst)
        
        params = {'objective' : trial.suggest_categorical('objective',param_grid['objective']),
                'metric' : trial.suggest_categorical('metric',param_grid['metric']),
                'verbosity' : trial.suggest_categorical('verbosity',param_grid['verbosity']),
                'boosting_type' : trial.suggest_categorical('boosting_type',param_grid['boosting_type']),
                'num_leaves' : trial.suggest_int('num_leaves', param_grid['num_leaves'][0], param_grid['num_leaves'][1]),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'lambda_l1' : trial.suggest_loguniform('lambda_l1', param_grid['lambda_l1'][0], param_grid['lambda_l1'][1]),
                'lambda_l2' : trial.suggest_loguniform('lambda_l2', param_grid['lambda_l2'][0], param_grid['lambda_l2'][1]),
                'feature_fraction' : trial.suggest_uniform('feature_fraction', param_grid['feature_fraction'][0], param_grid['feature_fraction'][1]),
                'bagging_fraction' : trial.suggest_uniform('bagging_fraction', param_grid['bagging_fraction'][0], param_grid['bagging_fraction'][1]),
                'bagging_freq' : trial.suggest_int('bagging_freq', param_grid['bagging_freq'][0], param_grid['bagging_freq'][1]),
                'min_child_samples' : trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0], param_grid['min_child_samples'][1]),
                'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1.0, classWeight])}
        
        return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)
	
	
def run_LGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est = lgb.LGBMClassifier()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_LGB(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = lgb.LGBMClassifier()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi
	
	
def objective_SVM(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
	params = {'kernel' : trial.suggest_categorical('kernel',param_grid['kernel']),
                'C' : trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
                'gamma' : trial.suggest_categorical('gamma',param_grid['gamma']),
                'degree' : trial.suggest_int('degree', param_grid['degree'][0], param_grid['degree'][1]),
                'probability' : trial.suggest_categorical('probability',param_grid['probability']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight'])}

	return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)
	
	
def run_SVM_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
	#Run Hyperparameter sweep
	est = SVC()
	sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
	study = optuna.create_study(direction='maximize', sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.CRITICAL)
	study.optimize(lambda trial: objective_SVM(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))

	fig = optuna.visualization.plot_parallel_coordinate(study)
	#fig.show()
	fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')
	
	print('Best trial:')
	best_trial = study.best_trial
	print('  Value: ', best_trial.value)
	print('  Params: ')
	for key, value in best_trial.params.items():
		print('    {}: {}'.format(key, value))
	
	#Train model using 'best' hyperparameters
	est = SVC()
	clf = clone(est).set_params(**best_trial.params)  
	setattr(clf,'random_state',randSeed)

	model = clf.fit(x_train, y_train)

	#Save model
	pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

	#Prediction evaluation
	y_pred = clf.predict(x_test)

	metricList = classEval(y_test, y_pred, False)

	#Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
	probas_ = model.predict_proba(x_test)

	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
	roc_auc = auc(fpr, tpr)

	# Compute Precision/Recall curve and AUC
	prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
	prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
	prec_rec_auc = auc(recall, prec)
	ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

	#Feature Importance Estimates
	fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

	return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

    
def objective_ANN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
        params = {'activation' : trial.suggest_categorical('activation',param_grid['activation']),
                'learning_rate' : trial.suggest_categorical('learning_rate',param_grid['learning_rate']),
                'momentum' : trial.suggest_uniform('momentum', param_grid['momentum'][0], param_grid['momentum'][1]),
                'solver' : trial.suggest_categorical('solver',param_grid['solver']),
                'batch_size' : trial.suggest_categorical('batch_size',param_grid['batch_size']),
                'alpha' : trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
                'max_iter' : trial.suggest_categorical('max_iter',param_grid['max_iter'])}
                
        n_layers = trial.suggest_int('n_layers', param_grid['n_layers'][0], param_grid['n_layers'][1])
        layers = []
        for i in range(n_layers):
                layers.append(trial.suggest_int('n_units_l{}'.format(i), param_grid['layer_size'][0], param_grid['layer_size'][1]))

        params['hidden_layer_sizes'] = tuple(layers)
                
        return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)
	
	
def run_ANN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,name_path, hype_cv, n_trials, scoring_metric,timeout,wd_path,output_folder,algorithm,data_name):
        #Run Hyperparameter sweep
        est = MLPClassifier()
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_ANN(trial,est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric), n_trials=n_trials, timeout=timeout, catch=(ValueError,))
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        #fig.show()
        fig.write_image(wd_path+output_folder+'/'+algorithm+'_hyperparams_' +data_name+'_'+ str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
                print('    {}: {}'.format(key, value))

        #Handle special parameter requirement for ANN
        layers = []
        for j in range(best_trial.params['n_layers']):
                layer_name = 'n_units_l'+str(j)
                layers.append(best_trial.params[layer_name])
                del best_trial.params[layer_name]

        best_trial.params['hidden_layer_sizes'] = tuple(layers)
        del best_trial.params['n_layers']

        #Train model using 'best' hyperparameters
        est = MLPClassifier()
        clf = clone(est).set_params(**best_trial.params)  
        setattr(clf,'random_state',randSeed)

        model = clf.fit(x_train, y_train)

        #Save model
        pickle.dump(model, open(name_path+str(i)+'.sav', 'wb'))

        #Prediction evaluation
        y_pred = clf.predict(x_test)

        metricList = classEval(y_test, y_pred, False)

        #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
        probas_ = model.predict_proba(x_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)

        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

        #Feature Importance Estimates
        fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

        return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi


def run_LCS_full(caseProbs, controlProbs, y_pred, y_test, i):
        metricList = classEval(y_test, y_pred, False)

        #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
        probas_ = calculateClassProbs(caseProbs,controlProbs)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)

        # Compute Precision/Recall curve and AUC
        prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
        prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
        prec_rec_auc = auc(recall, prec)
        ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

        return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec


def calculateClassProbs(caseProbs,controlProbs):
        probList = []
        for i in range(len(caseProbs)):
                voteSum = caseProbs[i]+controlProbs[i]
                caseProb = caseProbs[i]/float(voteSum)
                controlProb = controlProbs[i]/float(voteSum)
                probList.append([controlProb,caseProb])
    
        probas_ = np.asarray(probList)
    
        return probas_


def eval_LCS(algorithm,cv_partitions,wd_path,output_folder,lcs_folder,data_name,iterations,algColor):
        alg_result_table = []
        #Define evaluation stats variable lists
        s_bac = []
        s_ac = []
        s_f1 = []
        s_re = []
        s_sp = []
        s_pr = []
        s_tp = []
        s_tn = []
        s_fp = []
        s_fn = []

        #Define ROC plot variable lists
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        #Define PRC plot variable lists
        precs = []
        praucs = []
        aveprecs = []
        save_y_test = []
        for i in tqdm_notebook(range(cv_partitions), desc='1st loop'):
                #Algorithm Specific Code
                print("Running "+str(algorithm))

                #Load required results values
                resultFile = wd_path+lcs_folder+'/'+'ExSTraCS'+'_'+data_name+'_'+str(i)+'_Train_'+str(iterations)+'_Predictions.txt'
                df = pd.read_csv(resultFile, na_values='NA', sep = "\t")
                caseProbs = df['1'].values.tolist()
                controlProbs = df['0'].values.tolist()
                y_pred = df['Endpoint Predictions'].values.tolist()
                y_test = df['True Endpoint'].values.tolist()
                save_y_test = y_test
        
                metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec = run_LCS_full(caseProbs, controlProbs, y_pred, y_test,i)
        
                #Update evaluation stats variable lists [bac, ac, f1, re, sp, pr, tp, tn, fp, fn]
                s_bac.append(metricList[0])
                s_ac.append(metricList[1])
                s_f1.append(metricList[2])
                s_re.append(metricList[3])
                s_sp.append(metricList[4])
                s_pr.append(metricList[5])
                s_tp.append(metricList[6])
                s_tn.append(metricList[7])
                s_fp.append(metricList[8])
                s_fn.append(metricList[9])

                alg_result_table.append([fpr,tpr,roc_auc,recall,prec,prec_rec_auc,ave_prec])

                #Update ROC plot variable lists
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)

                #Update PRC plot variable lists
                precs.append(interp(mean_recall, recall, prec))
                praucs.append(prec_rec_auc)
                aveprecs.append(ave_prec)

        #ROC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
        plt.figure(figsize=(7,7))
        for i in range(cv_partitions):
                #Plot individual CV ROC line
                plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3,
                        label='ROC fold %d (AUC = %0.2f)' % (i, alg_result_table[i][2]))
        
        #Add chance line to ROC plot
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

        #Calculate and draw average CV ROC line
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=algColor,label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm+' : ROC over CV Partitions')
        plt.legend(loc="best")
        plt.savefig((wd_path+output_folder+'/'+algorithm+'_ROC_' + data_name), bbox_inches = "tight")
        plt.show()

        #PRC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
        plt.figure(figsize=(7,7))
        for i in range(cv_partitions):
                #Plot individual CV ROC line
                plt.plot(alg_result_table[i][3], alg_result_table[i][4], lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.2f) (APS = %0.2f)' % (i, alg_result_table[i][5],alg_result_table[i][6]))
        
        #Add chance line to PRC plot
        noskill = sum(save_y_test) / float(len(save_y_test)) #Fraction of cases
        plt.plot([0, 1], [noskill, noskill], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

        #Calculate and draw average CV PRC line
        mean_prec = np.mean(precs, axis=0)
        mean_pr_auc = auc(mean_recall, mean_prec)
        std_auc = np.std(praucs)
        plt.plot(mean_recall, mean_prec, color=algColor, label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_pr_auc, std_auc),lw=2, alpha=.8)

        std_prec = np.std(precs, axis=0)
        precs_upper = np.minimum(mean_prec + std_prec, 1)
        precs_lower = np.maximum(mean_prec - std_prec, 0)
        plt.fill_between(mean_fpr, precs_lower, precs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(algorithm+' : PRC over CV Partitions')
        plt.legend(loc="best")
        plt.savefig((wd_path+output_folder+'/'+algorithm+'_PRC_' + data_name), bbox_inches = "tight")
        plt.show()

        #Calculate and report average eval statistics.
        print("Avg. Model Balanced Accuracy = " + str(np.mean(s_bac)) + " (std. dev. = " + str(np.std(s_bac)) + ")")    
        print("Avg. Model Accuracy = " + str(np.mean(s_ac)) + " (std. dev. = " + str(np.std(s_ac)) + ")")
        print("Avg. Model F1-Score = " + str(np.mean(s_f1)) + " (std. dev. = " + str(np.std(s_f1)) + ")")
        print("Avg. Model Recall = " + str(np.mean(s_re)) + " (std. dev. = " + str(np.std(s_re)) + ")")
        print("Avg. Model Specificity = " + str(np.mean(s_sp)) + " (std. dev. = " + str(np.std(s_sp)) + ")")
        print("Avg. Model Precision = " + str(np.mean(s_pr)) + " (std. dev. = " + str(np.std(s_pr)) + ")")
        print("Avg. Model True Positives = " + str(np.mean(s_tp)) + " (std. dev. = " + str(np.std(s_tp)) + ")")
        print("Avg. Model True Negatives = " + str(np.mean(s_tn)) + " (std. dev. = " + str(np.std(s_tn)) + ")")
        print("Avg. Model False Positives = " + str(np.mean(s_fp)) + " (std. dev. = " + str(np.std(s_fp)) + ")")
        print("Avg. Model False Negatives = " + str(np.mean(s_fn)) + " (std. dev. = " + str(np.std(s_fn)) + ")")
        print("Avg. ROC AUC = " + str(np.mean(aucs)) + " (std. dev. = " + str(np.std(aucs)) + ")")
        print("Avg. PRC AUC = " + str(np.mean(praucs)) + " (std. dev. = " + str(np.std(praucs)) + ")")
        print("Avg. PRC Precision Score = " + str(np.mean(aveprecs)) + " (std. dev. = " + str(np.std(aveprecs)) + ")")
    
        # Save metrics for printing to file. 
        save_performance(algorithm,s_bac,s_ac,s_f1,s_re,s_sp,s_pr,s_tp,s_tn,s_fp,s_fn,aucs,praucs,aveprecs,output_folder,data_name)

        mean_ave_prec = np.mean(aveprecs)
        return mean_fpr, mean_tpr, mean_auc, mean_prec, mean_pr_auc, mean_ave_prec


def eval_LCS_QRF(algorithm,cv_partitions,wd_path,output_folder,lcs_folder,data_name,iterations,algColor):
        alg_result_table = []
        #Define evaluation stats variable lists
        s_bac = []
        s_ac = []
        s_f1 = []
        s_re = []
        s_sp = []
        s_pr = []
        s_tp = []
        s_tn = []
        s_fp = []
        s_fn = []

        #Define ROC plot variable lists
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        #Define PRC plot variable lists
        precs = []
        praucs = []
        aveprecs = []
        save_y_test = []
        for i in tqdm_notebook(range(cv_partitions), desc='1st loop'):
                #Algorithm Specific Code
                print("Running "+str(algorithm))

                #Load required results values
                resultFile = wd_path+lcs_folder+'/'+'ExSTraCS'+'_'+data_name+'_'+str(i)+'_Train_RC_QRF_'+str(iterations)+'_Predictions.txt'
                df = pd.read_csv(resultFile, na_values='NA', sep = "\t")
                caseProbs = df['1'].values.tolist()
                controlProbs = df['0'].values.tolist()
                y_pred = df['Endpoint Predictions'].values.tolist()
                y_test = df['True Endpoint'].values.tolist()
                save_y_test = y_test
        
                metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec = run_LCS_full(caseProbs, controlProbs, y_pred, y_test,i)
        
                #Update evaluation stats variable lists [bac, ac, f1, re, sp, pr, tp, tn, fp, fn]
                s_bac.append(metricList[0])
                s_ac.append(metricList[1])
                s_f1.append(metricList[2])
                s_re.append(metricList[3])
                s_sp.append(metricList[4])
                s_pr.append(metricList[5])
                s_tp.append(metricList[6])
                s_tn.append(metricList[7])
                s_fp.append(metricList[8])
                s_fn.append(metricList[9])

                alg_result_table.append([fpr,tpr,roc_auc,recall,prec,prec_rec_auc,ave_prec])

                #Update ROC plot variable lists
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)

                #Update PRC plot variable lists
                precs.append(interp(mean_recall, recall, prec))
                praucs.append(prec_rec_auc)
                aveprecs.append(ave_prec)

        #ROC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
        plt.figure(figsize=(7,7))
        for i in range(cv_partitions):
                #Plot individual CV ROC line
                plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, alg_result_table[i][2]))
        
        #Add chance line to ROC plot
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

        #Calculate and draw average CV ROC line
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=algColor, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm+' : ROC over CV Partitions')
        plt.legend(loc="best")
        plt.savefig((wd_path+output_folder+'/'+algorithm+'_ROC_' + data_name), bbox_inches = "tight")
        plt.show()

        #PRC plot (individual algorithm) -----------------------------------------------------------------------------------------------------
        plt.figure(figsize=(7,7))
        for i in range(cv_partitions):
                #Plot individual CV ROC line
                plt.plot(alg_result_table[i][3], alg_result_table[i][4], lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.2f) (APS = %0.2f)' % (i, alg_result_table[i][5],alg_result_table[i][6]))
        
        #Add chance line to PRC plot
        noskill = sum(save_y_test) / float(len(save_y_test)) #Fraction of cases
        plt.plot([0, 1], [noskill, noskill], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

        #Calculate and draw average CV PRC line
        mean_prec = np.mean(precs, axis=0)
        mean_pr_auc = auc(mean_recall, mean_prec)
        std_auc = np.std(praucs)
        plt.plot(mean_recall, mean_prec, color=algColor, label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_pr_auc, std_auc), lw=2, alpha=.8)

        std_prec = np.std(precs, axis=0)
        precs_upper = np.minimum(mean_prec + std_prec, 1)
        precs_lower = np.maximum(mean_prec - std_prec, 0)
        plt.fill_between(mean_fpr, precs_lower, precs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(algorithm+' : PRC over CV Partitions')
        plt.legend(loc="best")
        plt.savefig((wd_path+output_folder+'/'+algorithm+'_PRC_' + data_name), bbox_inches = "tight")
        plt.show()

        #Calculate and report average eval statistics.
        print("Avg. Model Balanced Accuracy = " + str(np.mean(s_bac)) + " (std. dev. = " + str(np.std(s_bac)) + ")")    
        print("Avg. Model Accuracy = " + str(np.mean(s_ac)) + " (std. dev. = " + str(np.std(s_ac)) + ")")
        print("Avg. Model F1-Score = " + str(np.mean(s_f1)) + " (std. dev. = " + str(np.std(s_f1)) + ")")
        print("Avg. Model Recall = " + str(np.mean(s_re)) + " (std. dev. = " + str(np.std(s_re)) + ")")
        print("Avg. Model Specificity = " + str(np.mean(s_sp)) + " (std. dev. = " + str(np.std(s_sp)) + ")")
        print("Avg. Model Precision = " + str(np.mean(s_pr)) + " (std. dev. = " + str(np.std(s_pr)) + ")")
        print("Avg. Model True Positives = " + str(np.mean(s_tp)) + " (std. dev. = " + str(np.std(s_tp)) + ")")
        print("Avg. Model True Negatives = " + str(np.mean(s_tn)) + " (std. dev. = " + str(np.std(s_tn)) + ")")
        print("Avg. Model False Positives = " + str(np.mean(s_fp)) + " (std. dev. = " + str(np.std(s_fp)) + ")")
        print("Avg. Model False Negatives = " + str(np.mean(s_fn)) + " (std. dev. = " + str(np.std(s_fn)) + ")")
        print("Avg. ROC AUC = " + str(np.mean(aucs)) + " (std. dev. = " + str(np.std(aucs)) + ")")
        print("Avg. PRC AUC = " + str(np.mean(praucs)) + " (std. dev. = " + str(np.std(praucs)) + ")")
        print("Avg. PRC Precision Score = " + str(np.mean(aveprecs)) + " (std. dev. = " + str(np.std(aveprecs)) + ")")
    
        # Save metrics for printing to file. 
        save_performance(algorithm,s_bac,s_ac,s_f1,s_re,s_sp,s_pr,s_tp,s_tn,s_fp,s_fn,aucs,praucs,aveprecs,output_folder,data_name)

        mean_ave_prec = np.mean(aveprecs)
        return mean_fpr, mean_tpr, mean_auc, mean_prec, mean_pr_auc, mean_ave_prec
