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

###################################################################################
###################################################################################

## dirección donde se encuentra el archivo fasta
## Carpeta: "fishes"
biol_dir = "/hpcfs/home/da.martinez33/Biologia"
jobPath = os.path.join(biol_dir,"results")
animal = "acyroltosyphon"
jobFile = animal + "_corr_sampling_progress.txt" # Archivo solo para ver el progreso del job

data_dir = os.path.join(biol_dir,"Data","insects",animal) #Carpeta donde está el archivo
print("Inciando con ({})....\n".format(animal))

for file in os.listdir(data_dir):
    if file.endswith(".fna"):

        file_dir = os.path.join(data_dir,file)
        print(file)

        # Aquí hace el corte de las secuencias en reads de 100 bp
        ###########################################################################
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            total_seqs_p_record = floor(len(seq_record)/seqs_size)
            # secuecias totales de 100 bp del registro
            get_100_seq(seq_record, seqs_size, total_seqs_p_record)

        print(len(fragmented_genome))
        ######## En esta parte selecciona solo 300.000 reads
        rws.write_job_results(jobPath,jobFile,len(subfragmented_genome)) # solo escribe el progreso del job
        shuffled = random.sample(range(len(fragmented_genome)), num_seqs_p_gen)
        
        for i in range(len(fragmented_genome)):
            if i in shuffled:
                subfragmented_genome.append(fragmented_genome[i])

        print(len(subfragmented_genome))
        rws.write_job_results(jobPath,jobFile,len(subfragmented_genome)) # solo escribe el progreso del job
        
        #### Se guardan los 300.000 reads en un archivo '.fasta'
        fragments_file = file[:-4] + "_sub_sampled_" + str(seqs_size) + ".fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        #### Si se quieren guardar todos los reads del genoma usar la variable 'fragmented_genome'
        SeqIO.write(subfragmented_genome, subsampled_path, "fasta")
        break
        ###########################################################################
    
    # Ignorar lo siguiete
    # Solo submuestrea otros archivos que ya estaban previamente fragmentados
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
        
        rws.write_job_results(jobPath,jobFile,len(subfragmented_genome))
        print(len(fragmented_genome))

        cont = 0
        for seq_record in SeqIO.parse(file_dir, "fasta"):
            if cont in shuffled:
                fragmented_genome.append(seq_record)
            cont += 1

        #######################################
        print(len(fragmented_genome))
        rws.write_job_results(jobPath,jobFile,len(subfragmented_genome))
        
        fragments_file = name_file + "_sub_sampled_" + str(seqs_size) + ".fasta"
        subsampled_path = os.path.join(data_dir,fragments_file)
        SeqIO.write(fragmented_genome, subsampled_path, "fasta")
        break
