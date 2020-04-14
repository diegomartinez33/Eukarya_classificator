#!/usr/bin/perl

# stickleback
echo "Stickleback"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/fishes/stickleback/GCA_006229165.1_NID_genomic_sub_sampled.fasta 2 > ./counts/stickleback_counts.txt

# whale-shark
echo "whale shark"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/fishes/whale_shark/GCF_001642345.1_ASM164234v2_genomic_sub_sampled.fasta 2 > ./counts/whaleshark_counts.txt

# Cod
echo "Cod"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/fishes/fragments/GR_Cod_sub_sampled.fasta 2 > ./counts/Cod_counts.txt

# Tilapia
echo "Tilapia"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/fishes/fragments/GR_Tilapia_sub_sampled.fasta 2 > ./counts/tilapia_counts.txt

# Salmon
echo "Salmon"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/fishes/fragments/GR_Salmon_sub_sampled.fasta 2 > ./counts/salmon_counts.txt

# acyroltosyphon
echo "acyroltosyphon"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/insects/acyroltosyphon/GCF_005508785.1_pea_aphid_22Mar2018_4r6ur_genomic_sub_sampled.fasta 2 > ./counts/acyroltosyphon_counts.txt

# bombix
echo "bombix"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/insects/bombix/GCF_000151625.1_ASM15162v1_genomic_sub_sampled.fasta 2 > ./counts/bombix_counts.txt

# harpegrathos
echo "harpegrathos"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/insects/harpegrathos/GCF_003227715.1_Hsal_v8.5_genomic_sub_sampled.fasta 2 > ./counts/harpegrathos_counts.txt

# tribolium
echo "tribolium"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/insects/tribolium/GCF_000002335.3_Tcas5.2_genomic_sub_sampled.fasta 2 > ./counts/tribolium_counts.txt

# locusta
echo "locusta"
perl GetNucFrequency_PerSeq_varK.pl /hpcfs/home/da.martinez33/Biologia/Data/insects/locusta/GCA_000516895.1_LocustGenomeV1_genomic_sub_sampled.fasta 2 > ./counts/locusta_counts.txt
