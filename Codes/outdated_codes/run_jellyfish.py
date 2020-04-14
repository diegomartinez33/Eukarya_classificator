import subprocess
import os

k_mers = [1,2,3]

bashCom_jelly = "jellyfish count -m 8 -s 1M -t 8 -C"
example = "jellyfish count -m 8 -s 1M -t 8 -C GR_Salmon_60simulado_clean.fasta -o 2mer_out_Salmon"
baseCommand = "jellyfish count -m {} -s 1M -t 8 -C {} -o {}"
dump_command = "jellyfish dump {} -o {}"
bashcom_histo = "jellyfish histo {} -o {}.histo"

## fishes
biol_dir = "/hpcfs/home/da.martinez33/Biologia"

data_dir = os.path.join(biol_dir,"Data","fishes")
print("Inciando con los peces....\n")
for animal in os.listdir(data_dir):
	print(animal)
	genome_dir = os.path.join(data_dir,animal)
	for file in os.listdir(genome_dir):
		if file.endswith("_sub_sampled.fasta"):
			file_dir = os.path.join(genome_dir,file)
			filename = file[:-6]
			for k in k_mers:
				infile = file_dir
				outfile = "{}/{}mer_out_{}".format(genome_dir,k,filename)
				outfile_counts = "{}/{}mer_counts_{}.fa".format(genome_dir,k,filename)
				command = baseCommand.format(k,infile,outfile)
				print(command)
				process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
				output, error = process.communicate()
				#Conteos
				counts_comm = dump_command.format(outfile,outfile_counts)
				print(counts_comm)
				process3 = subprocess.Popen(counts_comm.split(), stdout=subprocess.PIPE)
				output3, error3 = process3.communicate()
				# Histograma
				histo_comm = bashcom_histo.format(outfile,outfile)
				print(histo_comm)
				process2 = subprocess.Popen(histo_comm.split(), stdout=subprocess.PIPE)
				output2, error2 = process2.communicate()

## insects
data_dir = os.path.join(biol_dir,"Data","insects")
print("Inciando con los insectos....\n")
for animal in os.listdir(data_dir):
	print(animal)
	genome_dir = os.path.join(data_dir,animal)
	for file in os.listdir(genome_dir):
		if file.endswith("_sub_sampled.fasta"):
			file_dir = os.path.join(genome_dir,file)
			filename = file[:-6]
			for k in k_mers:
				infile = file_dir
				outfile = "{}/{}mer_out_{}".format(genome_dir,k,filename)
				outfile_counts = "{}/{}mer_counts_{}.fa".format(genome_dir,k,filename)
				command = baseCommand.format(k,infile,outfile)
				print(command)
				process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
				output, error = process.communicate()
				#Conteos
				counts_comm = dump_command.format(outfile,outfile_counts)
				print(counts_comm)
				process3 = subprocess.Popen(counts_comm.split(), stdout=subprocess.PIPE)
				output3, error3 = process3.communicate()
				# Histograma
				histo_comm = bashcom_histo.format(outfile,outfile)
				print(histo_comm)
				process2 = subprocess.Popen(histo_comm.split(), stdout=subprocess.PIPE)
				output2, error2 = process2.communicate()