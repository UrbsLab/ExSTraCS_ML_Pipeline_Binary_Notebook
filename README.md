# Introduction
This respository includes the code (including a Jupyter Notebook) to run the machine learning analysis pipeline (for binary classification) that pairs with our paper titled "A Rigorous Machine Learning Analysis Pipeline for Biomedical Binary Classification: Application in Pancreatic Cancer Nested Case-control Studies with Implications for Bias Assessments". This notebook presents an example of a 'rigorous' and well annotated machine learning (ML) analysis pipeline that could be reasonablly applied to various supervised learning classification tasks, but was developed here specifically for biomedical data mining/modeling. This pipeline could be used as follows:

* Apply/run pipeline (as is) on other binary classification datasets
* Utilize this pipeline as a blueprint to implement your own expanded or simplified anlaysis pipeline
* Utilize as an educational tool/example of how to run python-based data handling, preprocessing, feature selection, machine learning modeling, generating relevant visualizations, and running non-parametric statistical comparisons. Packages such as pandas, scikit-learn and scipy are used.
* Obtain the code to generate our porposed 'compound feature importance barplots'.

***
## Alternative Pipeline Code
This code represents our first ML analysis pipeline implementation as described in the paper above. It includes the ExSTraCS machine learning modeler which is a stand alone python implementation (outside of scikit-learn).  More recently we have implemented scikit-learn compatible versions of ExSTraCS and two other LCS algorithms (i.e. XCS, and eLCS). We have since implemented an updated/improved version of this ML analysis pipeline to include those three new scikit-learn compatible LCS algorithms along with nearest neighbors and gradient boosting classifiers.  For users looking to replicate the pipeline in the above paper, use this present repository. For those looking for the most up-to-date pipeline including 5 additional ML modeling algorithms we recommend using the very similar pipeline available at: https://github.com/UrbsLab/scikit_ML_Pipeline_Binary_Notebook

***
# Prerequisites for Use
## Environment Requirements
In order to run this pipeline as a Jupyter Notebook you must have the proper environment set up on your computer. Python 3 as well as a number of Python packages are required.  Most of these requirements are satisfied by installing the most recent version of anaconda (https://docs.anaconda.com/anaconda/install/). We used Anaconda3 with python version 3.7.7 during this pipeline development. In addition to the packages included in anaconda, the following packages will need to be installed separately (or possibly updated, if you have an older version installed):

* scikit-rebate (To install: pip install skrebate)
* xgboost (To install: pip install xgboost)
* lightgbm (To install: pip install lightgbm)
* optuna (To install: pip install optuna)

Additionally, while currently commented out in the file (modeling_methods.py) if you want the optuna hypterparameter sweep figures to appear within the jupyter notebook (via the command 'fig.show()' ) you will need to run the following installation commands.  This should only be required if you edit the python file to uncomment this line for any or all of the ML modeling algorithms. 

* pip install -U plotly>=4.0.0
* conda install -c plotly plotly-orca

Lastly, in order to include the stand-alone algorithm 'ExSTraCS' we needed to call this from the command line within this Jupyter Notebook.  As a result, the part of this notebook running ExSTraCS will only run properly if the path to the working directory used to run this notebook includes no spaces.  In other words if your path includes a folder called 'My Folder' vs. 'My_Folder' you will likely get a run error for ExSTraCS (at least on a Windows machine). Thus, make sure to check that wherever you are running this notebook from, that the entire path to the working directory does note include any spaces. 

***
## Dataset Requirements
This notebook loads a single dataset to be run through the entire pipeline. Here we summarize the requirements for this dataset:
* Ensure your data is in a single file: (If you have a pre-partitioned training/testing dataset, you should combine them into a single dataset before running this notebook)
* Any dataset specific cleaning, feature transformation, or feature engineering that may be needed in order to maximize ML performance should be conducted by the user separately or added to the beginning of this notebook. 
* The dataset should be in tab-delimited .txt format to run this notebook (as is).  Commented-out code to load a comma separated file (.csv) and excel file (.xlsx) is included in the notebook as an alternative. 
* Missing data values should be empty or indicated with an 'NA'.
* Dataset includes a header with column names. This should include a column for the binary class label and (optionally) a column for the instance ID, as well as columns for other 'features', e.g. independend variables. 
* The class labels should be 0 for the major class (i.e. the most frequent class), and 1 for the minor class.  This is important for generation of the precision/recall curve (PRC) plots. 
* This dataset is saved in the working directory containing the jupyter notebook file, and all other files in this repository.
* All variables in the dataset have been numerically encoded (otherwise additional data preprocessing may be needed)

***
# Usage
* First, ensure all of the environment and dataset requirments above are satisfied. 
* Next, save this repository to the desired 'working directory' on your pc (make sure there are no 'spaces' in the path to this directory!)
* Open the jupyter notebook file (https://jupyter.readthedocs.io/en/latest/running.html). We found that the most reliable way to do this and ensure your run environment is correct is to open the 'anaconda prompt' which comes with your anaconda installation.  Once opened type the command 'jupyter notebook'.  Then navigate to your working directory and click on the notebook file: 'Supervised_Classification_ML_Pipeline.ipynb'.
* Towards the beginning of the notebook in the section 'Set Dataset Pipeline Variables (Mandatory)', make sure to update your dataset-specific information (e.g. dataset name, outcome label, and instance label (if applicable)
* In the next notebook cell, 'Set Other Pipeline Variables (Optional)', you can 'optionally' set other analysis pipeline settings (e.g. number of cross validation partitions, what algorithms to include, etc)
* Next, in the next cell, 'ML Modeling Hyperparamters (Optional)' you can adjust the hyperparameters of all ML modeling algorithms to be explored in the respective hyperparameter sweeps. You can also adjust the overall optuna settings controlling the basics of how the hyperparameter sweeps are conducted. Note that 'adding' any other hyperparameters that have not been included in this section for a given ML modeler, will require updates to the code in the file 'modeling_methods.py'. We believe that we have included all critical run parameters for each ML algorithm so this should not be an issue for most users.
* Now that the code as been adapted to your desired dataset/analysis, click 'Kernel' on the Jupyter notebook GUI, and select 'Restart & Run All' to run the script.  
* Note that due to all that is involved in running this notebook, it may take several hours or more to complete running all analyses. Runtime can be shortened by picking a subset of ML algorithms, picking a smaller number of CV partitions, reducing 'n_trials' and 'hype_cv' which controls hyperparameter optimization, or reducing 'instanceSubset' which controls the maximum number of instances used to run Relief-based feature selection (note: these algorithms scale quadratically with number of training instances). 

***
# Repository Orientation
Included in this repository is the following: 
* The ML pipeline jupyter notebook, used to run the analysis - 'Supervised_Classification_ML_Pipeline.ipynb'
* An example/test dataset taken from the UCI repository - 'hcc-data_example.txt'
* A python script used by part 1 of the notebook - 'data_processing_methods.py'
* A python script used by part 2 of the notebook - 'feature_selection_methods.py'
* A python script used by part 3 of the notebook - 'modeling_methods.py'
* A folder containing python code for the ExSTraCS ML algorithm - 'exstracs_2.0.2.1_noclassmutate_lynch'
* A schematic summarizing the ML analysis pipeline - 'ML pipeline schematic2.png'

***
# Notebook Organization
## Part 1: Exploratory analysis, data cleaning, and creating n-fold CV partitioned datasets 
- Instances missing a class value are excluded
- The user can indicate other columns that should be excluded from the analysis
- The user can turn on/off the option to apply standard scaling to the data prior to CV partitioning or imputation
    - We use no scaling by default. This is because most methods should work properly without it, and in applying the model downstream, it is difficult to properly scale new data so that models may be re-applied later.
    - ANN modeling is sensitive to feature scaling, thus without it, performance not be as good. However this is only one of many challenges in getting ANN to perform well. 
- The user can turn on/off the option to impute missing values following CV partitioning
- The user can turn on/off the option for the code to automatically attempt to discriminate nominal from ordinal features
- The user can choose the number of CV partitions as well as the strategy for CV partitioning (i.e.  random (R), stratified (S), and matched (M) 
- CV training and testing datasets are saved as .txt files so that the same partitions may be analyzed external to this code
    
## Part 2: Feature selection
- The user can turn on/off the option to filter out the lowest scoring features in the data (i.e. to conduct not just feature importance evaluation but feature selection)
- Feature importance evaluation and feature selection are conducted within each respective CV training partition
- The pipeline reports feature importance estimates via two feature selection algorithms:
    - Mutual Information: Proficient at detecting univariate associations
    - MultiSURF: Proficient at detecting univariate associations, 2-way epistatic interactions, and heterogeneous associations
    
- When selected by the user, feature selection conservatively keeps any feature identified as 'potentially relevant' (i.e. score > 0) by either algorithm
- Since MultiSURF scales quadratically with the number of training instances, there is an option to utilize a random subset of instances when running this algorithm to save computational time.

## Part 3: Machine learning modeling
- Seven ML modeling algorithms have been implemented in this pipeline:
    - Logistic Regression (scikit learn)
    - Decision Tree (scikit learn)
    - Random Forest (scikit learn)
    - Naïve Bayes (scikit learn)
    - XGBoost (separate python package)
    - LightGBM (separate python package)
    - SVM (scikit learn)
    - ANN (scikit learn)
    - ExSTraCS (v2.0.2.1) - a Learning Classifier System (LCS) algorithm manually configured to run in this notebook
- User can select any subset of these methods to run
- ML modeling is conducted within each respective CV training partition on the respective feature subset selected within the given CV partition
- ML modeling begins with a hyperparameter sweep conducted with a grid search of hard coded run parameter options (user can edit as needed)
- Balanced accuracy is applied as the evaluation metric for the hyperparameter sweep

## Part 4: ML feature importance vizualization
Performs normalization and transformation of feature importances scores for all algorithms and generates our proposed 'compound feature importance plots'. 

***
## Schematic of ML Analysis Pipeline

<img src="ML pipeline schematic2.png" />

