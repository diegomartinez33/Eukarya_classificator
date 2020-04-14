import numpy as np
# import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import math

## fishes
biol_dir = "/hpcfs/home/da.martinez33/Biologia"

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1f%%' % (x)

formatter = FuncFormatter(thousands)

print("\n---------- Iniciar creacion de graficas de conteo de kmeros ----------\n")

type_organisms = "insects"

data_dir = os.path.join(biol_dir,"Data",type_organisms)
results_dir = os.path.join(biol_dir,"results",type_organisms)
print("Inciando con los peces....\n")
animal_count_1 = 0
animal_count_2 = 0
animal_count_3 = 0
stats_txt_1 = os.path.join(results_dir,"1_kmer_stats.txt")
stats_txt_2 = os.path.join(results_dir,"2_kmer_stats.txt")
stats_txt_3 = os.path.join(results_dir,"3_kmer_stats.txt")

if os.path.isfile(stats_txt_1):
	os.remove(stats_txt_1)
if os.path.isfile(stats_txt_2):
	os.remove(stats_txt_2)
if os.path.isfile(stats_txt_3):
	os.remove(stats_txt_3)

def write_histo(line,k_num):
	if k_num == 1:
		stats_txt = stats_txt_1
	elif k_num == 2:
		stats_txt = stats_txt_2
	else:
		stats_txt = stats_txt_3
	with open(stats_txt,'a') as f:
		f.write(line)

def get_name_file(filename):
	cont_ = 0
	get_two = False
	file_animal = ""
	for i in filename:
		if i == "_":
			cont_ += 1
		if cont_ == 4:
			return file_animal
			break
		if get_two:
			file_animal = file_animal + i
		if cont_ == 2:
			get_two = True
			continue

for animal in os.listdir(data_dir):
	print(animal)
	genome_dir = os.path.join(data_dir,animal)
	for file in os.listdir(genome_dir):
		if file.endswith(".fa"):
			file_dir = os.path.join(genome_dir,file)
			counts = []
			kmers = []
			with open(file_dir) as fp:
				for cnt, line in enumerate(fp):
					if math.fmod(cnt,2) == 0:
						counts.append(int(line[1:]))
					else:
						kmers.append(line)
					print("Line {}: {}".format(cnt, line))

			counts_np = np.asarray(counts)
			print(type(counts_np))
			counts_np = np.divide(counts_np,sum(counts_np)) * 100
			x = np.arange(len(kmers))

			line_to_write_1 = "{}\t"

			if file[0] == '1':
				if animal_count_1 == 0:
					line_to_write = line_to_write_1.format("Animal")
					for kmer in kmers:
						line_to_write = line_to_write + kmer[:-1] + "\t"
					line_to_write = line_to_write + "\n"
					write_histo(line_to_write,1)
				# Linea de conteo
				if animal == "fragments":
					Animal_name = get_name_file(file)
					line_to_write = line_to_write_1.format(Animal_name)
				else:
					line_to_write = line_to_write_1.format(animal)

				for i in counts_np:
					line_to_write = line_to_write + str(i) + "\t"
				line_to_write = line_to_write + "\n"
				write_histo(line_to_write,1)
				animal_count_1 += 1
			elif file[0] == '2':
				if animal_count_2 == 0:
					line_to_write = line_to_write_1.format("Animal")
					for kmer in kmers:
						line_to_write = line_to_write + kmer[:-1] + "\t"
					line_to_write = line_to_write + "\n"
					write_histo(line_to_write,2)
				# Linea de conteo
				if animal == "fragments":
					Animal_name = get_name_file(file)
					line_to_write = line_to_write_1.format(Animal_name)
				else:
					line_to_write = line_to_write_1.format(animal)

				for i in counts_np:
					line_to_write = line_to_write + str(i) + "\t"
				line_to_write = line_to_write + "\n"
				write_histo(line_to_write,2)
				animal_count_2 += 1
			else:
				if animal_count_3 == 0:
					line_to_write = line_to_write_1.format("Animal")
					for kmer in kmers:
						line_to_write = line_to_write + kmer[:-1] + "\t"
					line_to_write = line_to_write + "\n"
					write_histo(line_to_write,3)
				# Linea de conteo
				if animal == "fragments":
					Animal_name = get_name_file(file)
					line_to_write = line_to_write_1.format(Animal_name)
				else:
					line_to_write = line_to_write_1.format(animal)

				for i in counts_np:
					line_to_write = line_to_write + str(i) + "\t"
				line_to_write = line_to_write + "\n"
				write_histo(line_to_write,3)
				animal_count_3 += 1

			fig1, ax = plt.subplots()
			ax.yaxis.set_major_formatter(formatter)
			plt.bar(x, counts_np)
			plt.xticks(x, kmers)
			plt.xlabel('Kmers', labelpad=1)
			plt.ylabel('Counts')
			plt.title('Distribution for {}'.format(animal))

			fig1.savefig(genome_dir + '/' + file[0:-15] + '_histo.png')

			plt.close(fig1)
			print('Figure of {} plotted'.format(animal))