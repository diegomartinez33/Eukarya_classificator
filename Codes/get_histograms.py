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

data_dir = os.path.join(biol_dir,"Codes","counts")
results_dir = os.path.join(biol_dir,"results","hists")

for file in os.listdir(data_dir):
	if file.endswith(".txt"):
		file_name = file[:-4]
		print(file_name)
		file_dir = os.path.join(data_dir,file)
		labels = np.genfromtxt(file_dir, skip_header=1, usecols=0, dtype=str)
		raw_data = np.genfromtxt(file_dir, skip_header=1)[:,1:]
		print(raw_data.shape)
		print(raw_data[0].shape)
		r_trans = np.transpose(raw_data)
		print(r_trans.shape)
		print(r_trans[0].shape)
		print(r_trans[0])
		print(raw_data[0]) 
		sys.exit()
		data = {label: row for label, row in zip(labels, raw_data)}

		animal_dir = os.path.join(results_dir,file_name)
		if os.path.isdir(animal_dir) is not True:
			os.mkdir(animal_dir)

		for kmer in data.keys():
			counts = data[kmer]
			counts_np = np.asarray(counts)

			fig1, ax = plt.subplots()
			n, bins, patches = plt.hist(x=counts_np, bins='auto', 
				color='#0504aa')
			plt.grid(axis='y')
			plt.xlabel('Value')
			plt.ylabel('Frequency')
			plt.title(file_name + ' ' + kmer)

			save_file = kmer + '_histo.png'

			fig1.savefig(animal_dir + '/' + save_file)

			plt.close(fig1)
			print('Figure of {} for kmer {} plotted'
				.format(file_name,kmer))
