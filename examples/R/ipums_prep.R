## First time only: install both packages
if (TRUE) {
  install.packages("data.table", type = "source", repos = "http://Rdatatable.github.io/data.table")
  install.packages("MatrixModels", repos='http://cran.us.r-project.org')
}
require(data.table)   
require(MatrixModels) 

response <- "INCEARN"

## Read top 100k rows
## Download data from https://drive.google.com/file/d/0By-R0tLVMSykckNocHZPeEN1VU0/view?usp=sharing
DT <- fread("~/ipums_2000-2015.csv", nrows=100000)
print(ncol(DT))

## only keep rows with income of >100 dollars
DT = DT[INCEARN>100]

## drop highly correlated columns
DT[,c('HHINCOME','INCWAGE','INCTOT','FTOTINC',"INCBUS00",'INCSS',"INCWELFR","INCINVST", "INCRETIR", "INCSUPP", "INCOTHER"):=NULL] 
print(ncol(DT))

## Drop the CLUSTER column - is a categorical with > 30k factors
#DT[,CLUSTER:=as.factor(as.numeric(CLUSTER %% (max(CLUSTER)-min(CLUSTER)+1)))]
DT[,CLUSTER:=NULL] 

## label-encoding of categoricals (those cols with fewer than 1k cols, but not constant)
feature.names <- setdiff(names(DT), response)
for (ff in feature.names) {
  tt <- uniqueN(DT[[ff]])
  if (tt < 1000 && tt > 1) {
    DT[, (ff):=factor(DT[[ff]])]  
    print(paste0(ff,"has ",tt," levels"))
  }
  if (tt < 2) {
    print(paste0("dropping constant column: ", ff))
    DT[, (ff):=NULL]
  }
}
print(ncol(DT))

numCols <- names(DT)[which(sapply(DT, is.numeric))]
catCols <- names(DT)[which(sapply(DT, is.factor))]
print(paste0("numeric: ",     numCols))
print(paste0("categorical: ", catCols))

## impute missing values, drop near-const cols and standardize the data
cols <- setdiff(numCols,c(response))
for (c in cols) {
  DT[!is.finite(DT[[c]]), (c):=mean(DT[[c]], na.rm=TRUE)]
  if (!is.finite(sd(DT[[c]])) || sd(DT[[c]])<1e-4) 
    DT[,(c):=NULL]
  else
    DT[,(c):=scale(as.numeric(DT[[c]]))]
}
print(ncol(DT))

## one-hot encode the categoricals
DT2 <- as.data.table(model.matrix(DT[[response]]~., data = DT[,c(catCols), with=FALSE], sparse=FALSE))[,-1]
print(ncol(DT2))

## add back the numeric columns and assign back to DT
DT <- DT2[,(numCols):=DT[,numCols,with=FALSE]]

print(dim(DT))

## check validity of data
all(!is.na(DT))
all(sapply(DT, is.numeric))
all(sapply(DT, function(x) all(is.finite(x))))

## save preprocessed file as CSV
fwrite(DT, "train.csv")
