""" library file to ontain file names"""
import os
import sys
import numpy as np

ani_names = ["stickleback", "whaleshark","Cod","tilapia","salmon",
             "acyroltosyphon","bombix","harpegrathos","locusta","tribolium"] 
real_names = ["stickleback", "whale_shark","bacalao","tilapia","salmon",
             "Acyrthosiphon","bombyx","Harpegnathos","locusta","tribolium"]

def get_name_file(filename):
    cont_ = 0
    file_animal = ""
    for i in filename:
        if i == "_":
            cont_ += 1
        if cont_ == 2:
            return file_animal
            break
        file_animal = file_animal + i

def get_animal(file_str):
	animal_name = ""
	for char in file_str:
		if char != "_":
			animal_name = animal_name + char
		else:
			break
	return animal_name

def get_real_animal_name(filename):
    raw_name = get_animal(filename)
    real_name = ""
    for i in range(len(ani_names)):
        if raw_name == ani_names[i]:
            real_name = real_names[i]
            break
    return real_name