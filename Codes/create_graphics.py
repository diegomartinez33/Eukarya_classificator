import numpy as np
import os
import sys
import time
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser

# Code to append libraries
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
results_dir = os.path.join(biol_dir,"results","training")
sys.path.append(file_manage_folder)
classifiers_folder = os.path.join(biol_dir, 'Codes', 'classification')
sys.path.append(classifiers_folder)

from classification import train_nn_cpu

from file_management import rws_files as rws
from v_plots import animals_wrong_histo

type_crossval = 'k-fold'
type_classif = 'cnn'
others_cnn = '0.0001'
exp_folder = "{}_{}_{}_cv".format(type_crossval,type_classif,others_cnn)
savefolder = os.path.join(results_dir,exp_folder)
results_data = rws.loadData(savefolder + '/train_results.pkl')

for num_fold in range(len(results_data)):
    print("\n---------------- fold %d ---------------------\n" % num_fold)
    wrong_s_labels = results_data[num_fold]['wrong_results'][1]
    ids = results_data[num_fold]['wrong_results'][2]
    data_dir = os.path.join(biol_dir,"Data")
    fasta_file = ""
    wrong_seqs = []
    for idx,label in enumerate(wrong_s_labels):
        if label == "tribolium":
            directory = os.path.join(data_dir,'insects','tribolium')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "locusta":
            directory = os.path.join(data_dir,'insects','locusta')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "Harpegnathos":
            directory = os.path.join(data_dir,'insects','harpegrathos')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "bombyx":
            directory = os.path.join(data_dir,'insects','bombix')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "Acyrthosiphon":
            directory = os.path.join(data_dir,'insects','acyroltosyphon')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "whale_shark":
            directory = os.path.join(data_dir,'fishes','whale_shark')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "stickleback":
            directory = os.path.join(data_dir,'fishes','stickleback')
            for file in os.listdir(directory):
                if file.endswith("_genomic_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "salmon":
            directory = os.path.join(data_dir,'fishes','fragments')
            for file in os.listdir(directory):
                if file.endswith("_Salmon_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "tilapia":
            directory = os.path.join(data_dir,'fishes','fragments')
            for file in os.listdir(directory):
                if file.endswith("_Tilapia_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        elif label == "bacalao":
            directory = os.path.join(data_dir,'fishes','fragments')
            for file in os.listdir(directory):
                if file.endswith("_Cod_sub_sampled.fasta"):
                    fasta_file = os.path.join(directory,file)
        else:
            print(label)
            continue
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            if ids[idx] == str(seq_record.id):
                wrong_seqs.append(str(seq_record))
    save_file = os.path.join(savefolder, "fold_{}".format(num_fold),"wrong_seqs.fasta")
    #### Si se quieren guardar todos los reads del genoma usar la variable 'fragmented_genome'
    SeqIO.write(wrong_seqs, save_file, "fasta")        
    #print(ids)
    #save_file = os.path.join(exp_folder, "fold_{}".format(num_fold))
    #animals_wrong_histo(wrong_s_labels, save_file)