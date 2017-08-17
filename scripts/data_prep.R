#'Prep data for h2ogpuml
#'
#'@param data_path Path to data
#'@param response Response column as a string
#'@param output_csv_name Name of save csv file
#'@param output_feather_name Name of saved feather file
#'@export
prepData <- function(data_path="", response="", output_csv_name="data.csv", output_feather_name="data.feather"){
  wd <- getwd()
  print(paste0("CSV will be saved to current working directory -> ",wd))
  
  response <- response
  print(paste0("Response is -> ",response))
  
  print(paste0("Reading in -> ",data_path))
  DT <- fread(data_path)
  
  print("Number of columns:")
  print(ncol(DT))
  
  print("Number of rows:")
  print(nrow(DT))
  
  ## Label-encoding of categoricals (those cols with fewer than 1k levels, but not constant)
  print("Label encoding")
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
  
  print("Number of columns after label encoding:")
  print(ncol(DT))
  
  numCols <- names(DT)[which(sapply(DT, is.numeric))]
  catCols <- names(DT)[which(sapply(DT, is.factor))]
  print(paste0("Number of numeric columns: ",     length(numCols)))
  print(paste0("Number of categorical columns: ", length(catCols)))
  
  ## impute missing values, drop near-const cols and standardize the data
  print("Imputing missing values using mean")
  cols <- setdiff(numCols,c(response))
  for (c in cols) {
    DT[!is.finite(DT[[c]]), (c):=mean(DT[[c]], na.rm=TRUE)]
    if (!is.finite(sd(DT[[c]])) || sd(DT[[c]])<1e-4) 
      DT[,(c):=NULL]
    else
      DT[,(c):=scale(as.numeric(DT[[c]]))]
  }
  print("Number of columns after mean imputation:")
  print(ncol(DT))
  
  ## one-hot encode the categoricals
  print("One hot encoding data table categoricals only")
  DT2 <- as.data.table(model.matrix(DT[[response]]~., data = DT[,c(catCols), with=FALSE], sparse=FALSE))[,-1]
  print("Number of columns that have been one hot encoded:")
  print(ncol(DT2))
  
  ## add back the numeric columns and assign back to DT
  print("Add back numeric columns and assign to data table")
  DT <- DT2[,(numCols):=DT[,numCols,with=FALSE]]
  
  print("Final dimensions of data table after pre processing:")
  print(dim(DT))
  
  ## check validity of data
  all(!is.na(DT))
  all(sapply(DT, is.numeric))
  all(sapply(DT, function(x) all(is.finite(x))))
  
  ## save preprocessed file as CSV
  print(paste0("Saving processed data to ", wd))
  fwrite(DT, output_csv_name)
  
  #Write out with feather
  ## First time only: install both packages
  require(data.table)   
  require(feather)
  
  print("Reading in data that has been pre processed")
  DT <- fread(paste0(getwd(),"/",output_csv_name))
  
  print("Writing out feather file")
  write_feather(DT, output_feather_name)
}
