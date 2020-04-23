#!/bin/bash

#SBATCH --job-name=count_Kmers	#Nombre del job
#SBATCH -p long			#Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1				#Nodos requeridos, Default=1
#SBATCH -n 1				#Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=4		#Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=16gb			#Memoria en Mb por CPU, Default=2048
#SBATCH --time=2-10:00:00		#Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=clusterresults144@uniandes.edu.co
#SBATCH --mail-type=ALL			
#SBATCH -o kmers.o%j		#Nombre de archivo de salida

host=`/bin/hostname`
date=`/bin/date`
echo "Corriendo el sub-muestreo de genomas"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

module load jellyfish/2.2.6
cd /hpcfs/home/da.martinez33/Biologia/Codes/
python run_jellyfish.py
