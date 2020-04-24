""" library file to read and write some files"""
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
import os
import sys
import numpy as np
import pickle

def write_job_results(jobPath, jobFile,len_subfragmented):
    """ Function to write in a follow-job"""
    if os.path.isfile(os.path.join(jobPath, jobFile)):
        with open(os.path.join(jobPath, jobFile), 'a') as f:
            line = "sub size: " + str(len_subfragmented)
            f.write(line + "\n")
            f.close()
    else:
        with open(os.path.join(jobPath, jobFile), 'w') as f:
            line = "sub size: " + str(len_subfragmented)
            f.write(line + "\n")
            f.close()

def write_results(resultsPath, line_to_write):
    """ Function to write in a follow-job"""
    if os.path.isfile(resultsPath):
        with open(resultsPath, 'a') as f:
            f.write(line_to_write + "\n")
            f.close()
    else:
        with open(resultsPath, 'w') as f:
            f.write(line_to_write + "\n")
            f.close()

def loadData(dataPathFile):
    """ Function to load the database dictionary from input path of pickle """
    if dataPathFile[-3:] == 'pkl':
        dataBaseDict = pickle.load(open(dataPathFile, 'rb'))
        return dataBaseDict
    else:
        raise Exception('File that is trying to be loaded is not a pickle file\n')

def saveData(dataBase, savePathFile):
    """ Function to save the database dictionary from input path of pickle """
    if savePathFile[-3:] == 'pkl':
        pickle.dump(dataBase, open(savePathFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('Saved dataBase in: ', savePathFile)