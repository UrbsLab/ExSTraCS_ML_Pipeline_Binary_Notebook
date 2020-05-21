"""
Name:        ExSTraCS_Main.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     April 25, 2014
Modified:    August 25,2014
Description: This module is called to run ExSTraCS from the command line.  Initialization of the algorithm and key mechanisms takes place here.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V2.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks, 
such as biomedical/bioinformatics/epidemiological problem domains.  This algorithm should be well suited to any supervised learning problem involving 
classification, prediction, data mining, and knowledge discovery.  This algorithm would NOT be suited to function approximation, behavioral modeling, 
or other multi-step problems.  This LCS algorithm is most closely based on the "UCS" algorithm, an LCS introduced by Ester Bernado-Mansilla and 
Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS introduced by Stewart Wilson (1995).   

Copyright (C) 2014 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def exstracs_main(configurationFile):
    helpstr = """
    WARNING: Some error has caused ExSTraCS to fail.  Please ensure that a properly formatted configuration file is available, and the 
    respective file path has been given as an argument to ExSTraCS. Also ensure that you run parameters within the configuration file 
    have been set correctly.  If this fails to solve the issue, please examine standard out/error to track down the issue.
    """
    try: 
        #Initialize the Parameters object - this will parse the configuration file and store all constants and parameters.
        ConfigParser(configurationFile)
        if cons.offlineData:
            if cons.internalCrossValidation == 0 or cons.internalCrossValidation == 1:  #No internal Cross Validation
                #Engage Timer - tracks run time of algorithm and it's components.
                timer = Timer() #TIME
                cons.referenceTimer(timer)    
                cons.timer.startTimeInit()
                #Initialize the Environment object - this manages the data presented to ExSTraCS 
                env = Offline_Environment()
                cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
                cons.parseIterations()
    
                #Instantiate ExSTraCS Algorithm
                algorithm = ExSTraCS()
                if cons.onlyTest:
                    cons.timer.stopTimeInit()
                    algorithm.runTestonly()
                else:
                    if cons.onlyRC:
                        cons.timer.stopTimeInit()
                        algorithm.runRConly()
                    else: 
                        if cons.onlyEKScores:
                            cons.timer.stopTimeInit()
                            EK = ExpertKnowledge(cons)
                            print("Algorithm Run Complete")
                        else: #Run the ExSTraCS algorithm.
                            if cons.useExpertKnowledge: #Transform EK scores into probabilities weights for covering. Done once. EK must be externally provided.
                                cons.timer.startTimeEK()
                                EK = ExpertKnowledge(cons)
                                cons.referenceExpertKnowledge(EK)
                                #cons.makeExpert() #stores the calculated EK probabilities in ExSTraCS_Constants.
                                cons.timer.stopTimeEK()
                            
                            if cons.doAttributeTracking:
                                cons.timer.startTimeAT()
                                AT = AttributeTracking(True)
                                cons.timer.stopTimeAT()
                            else:
                                AT = AttributeTracking(False)
                            cons.referenceAttributeTracking(AT)
                            cons.timer.stopTimeInit()
                            algorithm.runExSTraCS()
                        
            else: #Run internal Cross Validation
                for part in range(cons.internalCrossValidation):
                    cons.updateFileNames(part)            
                    #Initialize new ExSTraCS run----------------------------------------------------------------------------------------
                    #Engage Timer - tracks run time of algorithm and it's components.
                    timer = Timer() #TIME
                    cons.referenceTimer(timer)    
                    cons.timer.startTimeInit()
                    
                    #Initialize the Environment object - this manages the data presented to ExSTraCS 
                    env = Offline_Environment()
                    cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
                    cons.parseIterations()
                     
                    #Instantiate ExSTraCS Algorithm
                    algorithm = ExSTraCS()
                    if cons.onlyTest:
                        cons.timer.stopTimeInit()
                        algorithm.runTestonly()
                    else:
                        if cons.onlyRC:
                            cons.timer.stopTimeInit()
                            algorithm.runRConly()
                        else: 
                            if cons.onlyEKScores:
                                cons.runFilter()
                                print("Algorithm Run Complete")
                            else: #Run the ExSTraCS algorithm.
                                if cons.useExpertKnowledge: #Transform EK scores into probabilities weights for covering. Done once. EK must be externally provided.
                                    cons.timer.startTimeEK()
                                    EK = ExpertKnowledge(cons)
                                    cons.referenceExpertKnowledge(EK)
                                    cons.timer.stopTimeEK()
                                    
                                if cons.doAttributeTracking:
                                    cons.timer.startTimeAT()
                                    AT = AttributeTracking(True)
                                    cons.timer.stopTimeAT()
                                else:
                                    AT = AttributeTracking(False)
                                cons.referenceAttributeTracking(AT)
                                cons.timer.stopTimeInit()
                                algorithm.runExSTraCS()
    
        else: #Online Dataset (Does not allow Expert Knowledge, Attribute Tracking, Attribute Feedback, or cross-validation)
            #Engage Timer - tracks run time of algorithm and it's components.
            timer = Timer() #TIME
            cons.referenceTimer(timer)
            cons.timer.startTimeInit()
            cons.overrideParameters()
            
            #Initialize the Environment object - this manages the data presented to ExSTraCS 
            env = Online_Environment() #Multiplexer is currently hard coded (Users may add code for different online environments of interest)
            cons.referenceEnv(env) #Send reference to environment object to constants - to access from anywhere in ExSTraCS
            cons.parseIterations() 
            
            #Instantiate ExSTraCS Algorithm
            cons.timer.stopTimeInit()
            algorithm = ExSTraCS()
            
            cons.timer.stopTimeInit()
            if cons.onlyRC:
                algorithm.runRConly()
            else: 
                algorithm.runExSTraCS()
            
    except Exception as ex:
        logging.exception(helpstr)
        sys.exit()
        
        
if __name__ == '__main__':
    #Import Required Modules-------------------------------
    import sys
    import logging
    from exstracs_timer import Timer
    from exstracs_configparser import ConfigParser
    from exstracs_offlineenv import Offline_Environment
    from exstracs_onlineenv import Online_Environment
    from exstracs_algorithm import ExSTraCS
    from exstracs_constants import *
    from exstracs_at import AttributeTracking
    from exstracs_ek import ExpertKnowledge
    #------------------------------------------------------
    configurationFile = sys.argv[1] 
    exstracs_main(configurationFile)
    