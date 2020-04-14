from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
import os
import sys
import numpy as np
import random
from math import floor
import time

num_seqs_p_gen = 300000
seqs_size = 100

#shuffled = random.sample(range(num_seqs), num_seqs_p_gen)

def write_job_results():
    """ Function to write in a follow-job"""
    if os.path.isfile(os.path.join(jobPath, jobFile)):
        with open(os.path.join(jobPath, jobFile), 'a') as f:
            line = "Correctes ids for " + animal
            f.write(line + "\n")
            f.close()
    else:
        with open(os.path.join(jobPath, jobFile), 'w') as f:
            line = "Correctes ids for " + animal
            f.write(line + "\n")
            f.close()

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

def change_id(ori_id,all_ids):
    """ Funcion to change repeated id seqs """
    new_id = ori_id
    counter = 0
    while new_id in all_ids:
        new_id = ori_id + "_" + str(counter + 1)
        counter += 1
    return new_id

## fishes
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
jobPath = os.path.join(biol_dir,"results")
animal = "stickleback"
jobFile = animal + "_corr_sampling_progress.txt"

data_dir = os.path.join(biol_dir,"Data","fishes",animal)
print("Inciando con ({})....\n".format(animal))
for file in os.listdir(data_dir):
    if file.endswith("genomic_sub_sampled.fasta"):
        file_dir = os.path.join(data_dir,file)
        print(file)

        seqs_ids = []
        corrected_seqs = []
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            #print(seq_record.id)
            seqs_ids.append(seq_record.id)

        #print(seqs_ids)
        print(len(seqs_ids))

        cont = 0
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            s_id = seq_record.id
            new_s_id = change_id(s_id,seqs_ids)
            new_seq = seq_record
            new_seq.id = new_s_id
            corrected_seqs.append(new_seq)
            seqs_ids[cont] = new_s_id
            cont += 1
            #time.sleep(10)

        print(len(seqs_ids),cont)
        for seq in seqs_ids:
            cont_s = 0
            for seq_1 in seqs_ids:
                if seq == seq_1:
                    cont_s += 1
            if cont_s > 1:
                print("There is an error\n")
                sys.exit()
                break

        write_job_results()
        
        fragments_file = animal + "_sub_sampled.fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        SeqIO.write(corrected_seqs, subsampled_path, "fasta")
        break
        
    if file.endswith("_sub_sampled.fasta"):
        name_file = get_name_file(file)
        file_dir = os.path.join(data_dir,file)
        print(file)
        seqs_ids = []
        corrected_seqs = []
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            seqs_ids.append(seq_record.id)

        print(len(seqs_ids))

        cont = 0
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            s_id = seq_record.id
            new_s_id = change_id(s_id,seqs_ids)
            new_seq = seq_record
            new_seq.id = new_s_id
            corrected_seqs.append(new_seq)
            seqs_ids[cont] = new_s_id
            cont += 1
            #time.sleep(10)

        print(len(seqs_ids),cont)
        for seq in seqs_ids:
            cont_s = 0
            for seq_1 in seqs_ids:
                if seq == seq_1:
                    cont_s += 1
            if cont_s > 1:
                print("There is an error\n")
                sys.exit()
                break

        write_job_results()
        
        fragments_file = name_file + "_corr_sub_sampled.fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        SeqIO.write(corrected_seqs, subsampled_path, "fasta")
        break
