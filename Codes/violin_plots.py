import numpy as np
import os
import sys
import math
import pandas as pd
import sys
import time
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Code for file managemente library
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
cv_library_folder = os.path.join(biol_dir,'Codes','cv_lib')
sys.path.append(file_manage_folder)
sys.path.append(cv_library_folder)

from file_management import get_names as gn
from file_management import rws_files as rws
from cv_lib import cv_ids

## Violin plots
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
ani_names = ["stickleback", "whaleshark","Cod","tilapia","salmon",
             "acyroltosyphon","bombix","harpegrathos","locusta","tribolium"] 
real_names = ["stickleback", "whale_shark","bacalao","tilapia","salmon",
             "Acyrthosiphon","bombyx","Harpegnathos","locusta","tribolium"]
filter_cvs = True
filter_value = 0.95
points = True
#List of animals

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                   ]
  
print("\n---------- Iniciar creacion de graficas de conteo de kmeros ----------\n")

data_dir = os.path.join(biol_dir,"Codes","counts")
results_dir = os.path.join(biol_dir,"results","violins_points")
if filter_cvs:
    results_dir = os.path.join(biol_dir,"results",
                               "violins" + "_" + str(filter_value))
    if points:
        results_dir = os.path.join(biol_dir,"results",
                               "violins_points")
    if os.path.isdir(results_dir) != True:
        os.mkdir(results_dir)

def get_95_perc_data(datalist):
    qinf = 0.025
    qsup = 0.975

    new_data = []
    for ani_data in datalist:
        qinf_value = np.quantile(ani_data,qinf)
        qsup_value = np.quantile(ani_data,qsup)

        selected_indx = np.where((ani_data > qinf_value)&(ani_data < qsup_value))[0]
        clear_data = np.delete(ani_data,selected_indx)
        new_data.append(clear_data)
        #sys.exit()
    return new_data


#Create dictionaries per animal
print("Creating animal dictionaries...\n")
animals = {} #Dictionary that contains counts for each animal
kmers = [] #List of kmers
cont = 0
for file in os.listdir(data_dir):
    if file.endswith(".txt"):	
        file_name = file[:-4]
        animal = gn.get_animal(file)
        print(file_name)
        file_dir = os.path.join(data_dir,file)
        labels = np.genfromtxt(file_dir, skip_header=1, usecols=0, dtype=str)
        raw_data = np.genfromtxt(file_dir, skip_header=1)[:,1:]
        if filter_cvs:
            real_name = gn.get_real_animal_name(file_name)
            tsv_file = "cv_allReads_" + real_name + ".tsv"
            cvs_file = os.path.join(data_dir,"Scripts","cv_results",tsv_file)
            raw_data = cv_ids.remove_outliers(raw_data,
                cv_ids.get_cv_idx(cvs_file,filter_value))
        data = {label: row for label, row in zip(labels, raw_data)}
        # mini dict to save counts per kmer (label)
        if cont == 0:
            kmers = labels
        animals[animal] = data
        cont += 1

#Separate per kmer
print("Creating dictionaries per each kmer...\n")
kmers_dict = {} #Dict that has freqs of all animals per kmer
#Keys are each kmer
for kmer in kmers:
	k_dict = {}
	animal_list = [] #List of animals per freq
	counts_list = [] #List of counts of that kmer, of that animal
	for anim in animals.keys():
		freqs = animals[anim][kmer]
		list_lenght = len(animals[anim][kmer])
		anim_rep_list = [anim] * list_lenght
		animal_list = animal_list + anim_rep_list
		#print(type(freqs))
		counts_list = np.concatenate((np.asarray(counts_list),freqs), 
			axis=None)
	k_dict['animal'] = animal_list #Key for animal names
	k_dict['freq'] = counts_list #Key for freqs of each animal
	kmers_dict[kmer] = k_dict

#Create violin plots
print("\nCreating violing plots...\n")
print(animals.keys())
print(ani_names)

# for kmer in kmers:
#     fig, axes = plt.subplots()
#     frame_data = kmers_dict[kmer]
#     df = pd.DataFrame(frame_data)
#     kmer_data = []
#     for ani in ani_names:
#         kmer_data.append(df[df.animal == ani]['freq'].values)
    
#     axes.violinplot(dataset =kmer_data)
#     axes.set_title(kmer + ' kmer freqs')
#     axes.yaxis.grid(True)
#     axes.set_xlabel('Animal')
#     axes.set_ylabel('Frequency')
#     axes.set_xticks(np.arange(1, len(ani_names) + 1))
#     axes.set_xticklabels(real_names)
#     plt.setp(axes.get_xticklabels(), ha="right", rotation=45)

# 	#save image
#     plot_file = kmer + "_violin_plot.png"
#     print(plot_file)
#     plot_save_dir = os.path.join(results_dir,plot_file)
#     fig.savefig(plot_save_dir, bbox_inches="tight")
#     plt.close(fig)
#     #sys.exit()

for kmer in kmers:
    fig,axes = plt.subplots()
    frame_data = kmers_dict[kmer]
    df = pd.DataFrame(frame_data)
    kmer_data = []
    for ani in ani_names:
        kmer_data.append(df[df.animal == ani]['freq'].values)
    
    kmer_data = np.asarray(kmer_data)
    points_data = get_95_perc_data(kmer_data)
    #print(kmer_data.shape)
    print(kmer_data[0].shape)
    print(points_data[0].shape)
    #time.sleep(20)

    sns.violinplot(data=kmer_data, inner=None, palette=pkmn_type_colors)
    #sns.swarmplot(data=points_data, color='k', alpha=0.7)
    sns.stripplot(data=points_data, color='k', alpha=0.5, jitter=0.0001)
    axes.set_title(kmer + ' kmer freqs')
    axes.yaxis.grid(True)
    axes.set_xlabel('Animal')
    axes.set_ylabel('Frequency')
    axes.set_xticks(np.arange(1, len(ani_names) + 1))
    axes.set_xticklabels(real_names)
    plt.setp(axes.get_xticklabels(), ha="right", rotation=45)

    #save image
    plot_file = kmer + "_swarm_violin_plot.png"
    print(plot_file)
    plot_save_dir = os.path.join(results_dir,plot_file)
    fig.savefig(plot_save_dir, bbox_inches="tight")
    plt.close(fig)
    #sys.exit()