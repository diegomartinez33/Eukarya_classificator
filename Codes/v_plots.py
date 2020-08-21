import numpy as np
import os
import sys
import math
import pandas as pd
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Code for file managemente library
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
sys.path.append(file_manage_folder)

from file_management import rws_files as rws

biol_dir = "/hpcfs/home/da.martinez33/Biologia"
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

def get_viols_wrongs(wrong_data, wrong_s_labels, kmers, save_folder, all_data=False):
    """ Function to plot data misclassifed in total data distribution """

    print("\n---------- Iniciar creacion de graficas de conteo de kmeros ----------\n")

    results_dir_violins = os.path.join(biol_dir,"results","wrong_clf_violins", save_folder)
    if os.path.isdir(results_dir_violins) != True:
        os.makedirs(results_dir_violins)

    def create_dictionaries(kmers_list, animals_dict):
        print("Creating dictionaries per each kmer...\n")
        kmers_dict = {} #Dict that has freqs of all animals per kmer
        #Keys are each kmer
        for kmer in kmers_list:
            k_dict = {}
            animal_list = [] #List of animals per freq
            counts_list = [] #List of counts of that kmer, of that animal
            for anim in animals_dict.keys():
                freqs = animals_dict[anim][kmer]
                list_lenght = len(animals_dict[anim][kmer])
                anim_rep_list = [anim] * list_lenght
                animal_list = animal_list + anim_rep_list
                #print(type(freqs))
                counts_list = np.concatenate((np.asarray(counts_list),freqs), 
                    axis=None)
            k_dict['animal'] = animal_list #Key for animal names
            k_dict['freq'] = counts_list #Key for freqs of each animal
            kmers_dict[kmer] = k_dict
        return kmers_dict

    def create_wrong_dictionaries(kmers_list, data, s_labels):
        print("Creating dictionaries per each kmer...\n")
        kmers_dict = {}
        counts = np.transpose(data)
        for i, kmer in enumerate(kmers_list):
            k_dict = {}
            k_dict['animal'] = s_labels
            k_dict['freq'] = counts[i,:]
            kmers_dict[kmer] = k_dict
        return kmers_dict


    #Create dictionaries per animal
    print("Reading animal dictionaries...\n")
    animals = {} #Dictionary that contains counts for each animal

    wrong_data_dict = create_wrong_dictionaries(kmers, wrong_data, wrong_s_labels)

    if all_data:
        results_dir = os.path.join(biol_dir,"Data","classification")
        raw_data_file = os.path.join(results_dir,"counts_per_animal.pkl")
        counts_data = rws.loadData(raw_data_file)

        for animal in real_names:
            dict_animal = counts_data[animal]
            data = dict_animal['data']
            raw_data = np.transpose(data)
            data = {label: row for label, row in zip(kmers, raw_data)}
            animals[animal] = data

        all_data_dict = create_dictionaries(kmers, animals)

        #Create violin plots
        print("\nCreating violing plots...\n")

        for kmer in kmers:
            fig,axes = plt.subplots()
            frame_data = all_data_dict[kmer]
            w_frame_data = wrong_data_dict[kmer]
            df = pd.DataFrame(frame_data)
            w_df = pd.DataFrame(w_frame_data)
            
            kmer_data = []
            for ani in real_names:
                kmer_data.append(df[df.animal == ani]['freq'].values)
            kmer_data = np.asarray(kmer_data)
            
            points_data = []
            for ani in real_names:
                points_data.append(w_df[w_df.animal == ani]['freq'].values)
            points_data = np.asarray(points_data)

            #print(kmer_data.shape)
            #print(kmer_data[0].shape)
            #print(points_data[0].shape)
            #time.sleep(20)

            sns.violinplot(data=kmer_data, inner=None, palette=pkmn_type_colors)
            #sns.swarmplot(data=points_data, color='k', alpha=0.7)
            sns.stripplot(data=points_data, color='r', alpha=0.5, jitter=0.0001)
            axes.set_title(kmer + ' kmer freqs')
            axes.yaxis.grid(True)
            axes.set_xlabel('Animal')
            axes.set_ylabel('Frequency')
            axes.set_xticks(np.arange(1, len(real_names) + 1))
            axes.set_xticklabels(real_names)
            plt.setp(axes.get_xticklabels(), ha="right", rotation=45)

            #save image
            plot_file = kmer + "_all_and_missclass_violin_plot.png"
            print(plot_file)
            plot_save_dir = os.path.join(results_dir_violins,plot_file)
            fig.savefig(plot_save_dir, bbox_inches="tight")
            plt.close(fig)
    else:
        #Create violin plots
        print("\nCreating violing plots...\n")

        for kmer in kmers:
            fig,axes = plt.subplots()
            w_frame_data = wrong_data_dict[kmer]
            w_df = pd.DataFrame(w_frame_data)

            points_data = []
            for ani in real_names:
                points_data.append(w_df[w_df.animal == ani]['freq'].values)
            points_data = np.asarray(points_data)

            #print(points_data[0].shape)

            sns.violinplot(data=points_data, inner=None, palette=pkmn_type_colors)
            axes.set_title(kmer + ' kmer freqs')
            axes.yaxis.grid(True)
            axes.set_xlabel('Animal')
            axes.set_ylabel('Frequency')
            axes.set_xticks(np.arange(1, len(real_names) + 1))
            axes.set_xticklabels(real_names)
            plt.setp(axes.get_xticklabels(), ha="right", rotation=45)

            #save image
            plot_file = kmer + "_missclass_violin_plot.png"
            print(plot_file)
            plot_save_dir = os.path.join(results_dir_violins,plot_file)
            fig.savefig(plot_save_dir, bbox_inches="tight")
            plt.close(fig)
