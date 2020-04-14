#Activate arguments reading
args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}

cv_file=args[1]
counts_filem=args[2]

cv_matrix<-read.table(cv_file,h=T)

##Revisar distribución de los datos. escoger percentil de outliers
#quantile(cv_organism)
##Cuáles filas (índices) del vector (Bacteria[,17]) tiene un valor >= a lo que es
##considerado outlier
ind<-which(cv_matrix>=quantile(cv_matrix,probs = 0.9))
##Remover outliers de todo el dataset
#Bacteria1[-which(Bacteria1[,17]>=quantile(Bacteria1[,17],probs = 0.9)),]


