""" library file to manage data related to Variation Coefficient"""
import pandas as pd
import os
import sys
import numpy as np

def get_cv_idx(tvs_file,filter_value):
    cvs=pd.read_csv(tvs_file)
    numpy_cvs = cvs.values
    if filter_value < 1:
        quantile_percentage = filter_value
        quantile_value = cvs.quantile(quantile_percentage, axis=0)
        quantile_value = float(quantile_value)
        cvs_idx = np.asarray(np.where(numpy_cvs > quantile_value)[0])
    else:
        cvs_idx = np.asarray(np.where(numpy_cvs > filter_value)[0])
    return cvs_idx

def remove_outliers(data_matrix,cvs_idxs):
    clean_counts = np.delete(data_matrix,cvs_idxs,axis=1)
    return clean_counts