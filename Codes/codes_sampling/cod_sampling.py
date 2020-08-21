from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
import os
import sys
import numpy as np
import random
from math import floor
import time

num_seqs_p_gen = 300000
seqs_size = 200

# Importar librería para el manejo de archivos
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
file_manage_folder = os.path.join(biol_dir,'Codes','file_management')
sys.path.append(file_manage_folder)

#Ignorar estas librerias
import get_names as gn
import rws_files as rws

#shuffled = random.sample(range(num_seqs), num_seqs_p_gen)

################# Estas dos variables son las más importantes ####################

fragmented_genome = [] # Lista donde se almacenan los reads de 100 bp del genoma
subfragmented_genome = [] # Lista donde se almacenan 300000 reads de 100 bp

##################################################################################
######################### Funciones para cortar las secuencias ###################

def get_100_seq(sequence,seq_size,num_seqs_p_record):
    """ 
    Function which gets 100bp fragments from contigs sequences
    sequence: secuencia (contig)
    seq_size: seqs_size = 100
    num_seqs_p_record: number of fragments that will result after
    fragmenting the contig
    final_seqs: number of final fragments selected"""

    for i in range(num_seqs_p_record):
        ini = i * seq_size
        fin = (i + 1) * seq_size
        sub_seq = sequence[ini:fin]
        sub_seq.id = sub_seq.id + "_" + str(i) #Cambia el id del nuevo read
        if if_N_seq(sub_seq): #Mira si es una secuencia con muchas 'N'
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

###################################################################################
###################################################################################

## dirección donde se encuentra el archivo fasta
## Carpeta: "fishes"
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
jobPath = os.path.join(biol_dir,"results")
animal = "fragments"
jobFile = "Cod" + "_corr_sampling_progress.txt" # Archivo solo para ver el progreso del job

data_dir = os.path.join(biol_dir,"Data","fishes",animal) #Carpeta donde está el archivo
print("Inciando con ({})....\n".format(animal))

file = "GR_Cod_60simulado_clean.fasta"
      
if file.endswith("clean.fasta"):
    name_file = gn.get_name_file(file)
    file_dir = os.path.join(data_dir,file)
    print(file)
    print("Contar secuencias por genoma...")
    #######################################
    records = SeqIO.parse(file_dir, "fasta")
    num_seqs = len(list(records))
    seqs_p_record = 1
    # numero de secuencias por registro para completar 300.000
    print("numero de secuencias: " + str(num_seqs))

    cont = 0
    shuffled = random.sample(range(num_seqs), num_seqs_p_gen)
    
    rws.write_job_results(jobPath,jobFile,len(fragmented_genome))
    print(len(fragmented_genome))

    cont = 0
    for seq_record in SeqIO.parse(file_dir, "fasta"):
        if cont in shuffled:
            fragmented_genome.append(seq_record)
        cont += 1

    #######################################
    print(len(fragmented_genome))
    rws.write_job_results(jobPath,jobFile,len(fragmented_genome))
    
    fragments_file = name_file + "_sub_sampled_" + str(seqs_size) + ".fasta"
    subsampled_path = os.path.join(data_dir,fragments_file)
    SeqIO.write(fragmented_genome, subsampled_path, "fasta")
