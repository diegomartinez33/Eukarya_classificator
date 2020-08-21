#Activate arguments reading
args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

raw_matrix=args[1]
organism=args[2]
count_value<-0.1

library(data.table)

#Read counts in a data frame
matrix<-fread(raw_matrix)
matrix<-setDF(matrix)

CV<-function(count_Matrix){
  cv<-rep(0,ncol(count_Matrix)-1)
  for(i in 2:ncol(count_Matrix)){
    cv[i-1]<-sd(count_Matrix[,i])/mean(count_Matrix[,i])
  }
  return(cv)
}

GG_data<-function(dataframe){
	gg<-rep(0,ncol(dataframe)-1)
	for(i in 2:ncol(dataframe)){
		gg[i-1]<-as.numeric(dataframe[1,i])
	}
	return(gg)
}

print("a huevo que si")
#Get freqs for GG
GG_df<-matrix[matrix$Matrix=="GG",]
#GG_counts<-as.numeric(split(GG_df[,2:ncol(GG_df)], seq(ncol(GG_df)-1)))
#GG_counts<-GG_data(GG_df)
GG_counts<-as.numeric(GG_df[1,2:ncol(GG_df)])
print(length(GG_counts))

# cont_2<-0
# for (i in 2:ncol(matrix)){
#   if (GG_df[1,i] != matrix[20,i]){
#     break
#   }
# }

indx<-which(GG_counts > count_value & GG_counts < 1.1)
print(length(indx))

if (length(indx) >= 200){
	indx_200<-sample(indx,200)
} else {
	indx_200 <- indx
}

cont<-0
for (i in indx_200){
  if (GG_counts[i] > 0.1){
    cont<-cont+1
  }
}

print(cont)

indx_200_s <- indx_200 + 1
indx_200_s <- append(c(1),indx_200_s)

cv_organism<-CV(matrix)
sample_matrix<-matrix[,indx_200_s]
print(nrow(sample_matrix))
print(ncol(sample_matrix))

# pdf(paste("./GG_0.1/",organism,'_CV_GG_0.1.pdf',sep=""))
# hist(cv_organism[indx_200], col = "royalblue2", main =organism,
#      xlab = "CV", ylab = "Frequency")
# dev.off()

#print("Seleccion de frecuencias de muestra de 200")
print(paste0("generate CVs for: ", organism, " 200bp sequences"))
filename=paste("./cv_results_200bp/cv_allReads_",organism,".tsv",sep="")
write.table(cv_organism,filename,sep="\t",row.names=F,col.names=T)
pdf(paste("./cv_hists/",organism,'_200bp_CV_distr.pdf',sep=""))
hist(cv_organism, col = "royalblue2", main =organism,
     xlab = "CV", ylab = "Frecuencia")
dev.off()

#filename=paste("./sample_200_counts/",organism,"_sample.tsv",sep="")
#print(filename)
#write.table(sample_matrix,filename,sep="\t",row.names=F,col.names=T)

##Filtrar
##Revisar distribución de los datos. escoger percentil de outliers
#quantile(cv_organism)
##Cuáles filas (índices) del vector (Bacteria[,17]) tiene un valor >= a lo que es
##considerado outlier
#which(Bacteria1[,17]>=quantile(Bacteria1[,17],probs = 0.9))
##Remover outliers de todo el dataset
#Bacteria1[-which(Bacteria1[,17]>=quantile(Bacteria1[,17],probs = 0.9)),]

