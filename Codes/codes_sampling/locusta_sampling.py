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

fragmented_genome = []
subfragmented_genome = []

def get_100_seq(sequence,seq_size,num_seqs_p_record):
    """ 
    Function which gets 100bp fragments from contigs sequences
    sequence: secuencia (contig)
    seq_size: seqs_size = 100
    num_seqs_p_record: number of fragments that will result after
    fragmenting the contig
    final_seqs: number of final fragments selected"""
    cutted_sequences = []
    final_list = []

    for i in range(num_seqs_p_record):
        ini = i * seq_size
        fin = (i + 1) * seq_size
        sub_seq = sequence[ini:fin]
        sub_seq.id = sub_seq.id + "_" + str(i)
        if if_N_seq(sub_seq):
        	continue
        else:
        	fragmented_genome.append(sub_seq)

def if_N_seq(sequence):
	""" Function to check if a sequence is an 'N' sequence """
	max_count = 10 #### Parametro para eliminar una secuencia
	n_count = 0
	boolean = False
	for base in sequence:
	    if str(base) == "N":
	        n_count += 1
	    if n_count >= max_count:
	    	boolean = True
	    	break
	return boolean

def write_job_results():
    """ Function to write in a follow-job"""
    if os.path.isfile(os.path.join(jobPath, jobFile)):
        with open(os.path.join(jobPath, jobFile), 'a') as f:
            line = "sub size: " + str(len(subfragmented_genome))
            f.write(line + "\n")
            f.close()
    else:
        with open(os.path.join(jobPath, jobFile), 'w') as f:
            line = "sub size: " + str(len(subfragmented_genome))
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

## fishes
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
jobPath = os.path.join(biol_dir,"results")
animal = "locusta"
jobFile = animal + "_corr_sampling_progress.txt"

data_dir = os.path.join(biol_dir,"Data","insects",animal)
print("Inciando con ({})....\n".format(animal))
#for animal in os.listdir(data_dir):
for file in os.listdir(data_dir):
    if file.endswith(".fna"):
        file_dir = os.path.join(data_dir,file)
        print(file)
        #print("Contar secuencias por genoma...")
        #num_seqs = 0
        #with open(file_dir) as in_handle:
        #    for title, seq in SimpleFastaParser(in_handle):
        #        num_seqs += 1

        #cont = 0
        #with open(file_dir) as in_handle:
        #    for title, seq in SimpleFastaParser(in_handle):
        #        if cont in shuffled:
        #            sampled_seqs.append([seq])
        #        cont += 1

        for seq_record in SeqIO.parse(file_dir, "fasta"):
                total_seqs_p_record = floor(len(seq_record)/seqs_size)
                # secuecias totales de 100 bp del registro
                record_seqs = get_100_seq(seq_record,seqs_size,
                    total_seqs_p_record)

        print(len(fragmented_genome))
        write_job_results()
        #time.sleep(5)
        shuffled = random.sample(range(len(fragmented_genome)), num_seqs_p_gen)
        
        #TODO: code to eliminate or select the random-shuffled fragments
        for i in range(len(fragmented_genome)):
            if i in shuffled:
                subfragmented_genome.append(fragmented_genome[i])

        print(len(subfragmented_genome))
        write_job_results()
        
        fragments_file = file[:-4] + "_sub_sampled.fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        SeqIO.write(subfragmented_genome, subsampled_path, "fasta")
        break
        
    if file.endswith("clean.fasta"):
        name_file = get_name_file(file)
        file_dir = os.path.join(data_dir,file)
        print(file)
        print("Contar secuencias por genoma...")
        records = SeqIO.parse(file_dir, "fasta")
        num_seqs = len(list(records))
        #print(num_seqs)
        seqs_p_record = 1
        # numero de secuencias por registro para completar 300.000
        print("numero de secuencias: " + str(num_seqs))

        cont = 0
        shuffled = random.sample(range(num_seqs), num_seqs_p_gen)
        
        write_job_results()
        print(len(fragmented_genome))

        cont = 0
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            if cont in shuffled:
                fragmented_genome.append(seq_record)
                #print(len(list(seq_record)))
            cont += 1

        print(len(fragmented_genome))
        write_job_results()
        #time.sleep(5)
        #shuffled = random.sample(range(len(fragmented_genome)), num_seqs_p_gen)
        
        #TODO: code to eliminate or select the random-shuffled fragments
        #for i in range(len(fragmented_genome)):
        #    if i in shuffled:
        #        subfragmented_genome.append(fragmented_genome[i])

        #print(len(subfragmented_genome))
        
        fragments_file = name_file + "_sub_sampled.fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        SeqIO.write(fragmented_genome, subsampled_path, "fasta")
        break
