#'Prep data for h2o4gpu algorithms
#'
#'@param data_table `data.table` object containing data that needs to be preprocessed for h2o4gpu
#'@param response Response column as a string or index
#'@param save_csv_path Path to save processed data as a csv
#'@export
prep_data <- function(data_table, response, save_csv_path = NULL){
  
  if (!is.data.table(data-table)) {
    stop ("Input data should be of type data.table")
  }
  
  if (is.character(response)) {
    print(paste0("Response is -> ",response))
  } else {
    print(paste0("Response is -> ",colnames(data_table)[response]))
  }
  
  print(paste0("Number of columns: ", ncol(data_table)))
  
  print(paste0("Number of rows: ", nrow(data_table)))

  ## Label-encoding of categoricals (those cols with fewer than 1k levels, but not constant)
  print("Label encoding dataset...")
  feature.names <- setdiff(names(data_table), response)
  for (ff in feature.names) {
    tt <- uniqueN(data_table[[ff]])
    if (tt < 1000 && tt > 1) {
      data_table[, (ff):=factor(data_table[[ff]])]  
      print(paste0(ff," has ",tt," levels"))
    }
    if (tt < 2) {
      print(paste0("Dropping constant column: ", ff))
      data_table[, (ff):=NULL]
    }
  }
  
  print(paste0("Number of columns after label encoding: ", ncol(data_table)))
  
  numCols <- names(data_table)[which(sapply(data_table, is.numeric))]
  catCols <- names(data_table)[which(sapply(data_table, is.factor))]
  print(paste0("Number of numeric columns: ", length(numCols)))
  print(paste0("Number of categorical columns: ", length(catCols)))
  
  ## impute missing values, drop near-const cols and standardize the data
  print("Imputing missing values using mean...")
  cols <- setdiff(numCols,c(response))
  for (c in cols) {
    data_table[!is.finite(data_table[[c]]), (c):=mean(data_table[[c]], na.rm=TRUE)]
    if (!is.finite(sd(data_table[[c]])) || sd(data_table[[c]])<1e-4) 
      data_table[,(c):=NULL]
    else
      data_table[,(c):=scale(as.numeric(data_table[[c]]))]
  }
  print(paste0("Number of columns after mean imputation: ", ncol(data_table)))
  
  ## one-hot encode the categoricals
  print("One hot encoding data table categoricals only...")
  data_table2 <- as.data.table(model.matrix(data_table[[response]]~., data = data_table[,c(catCols), with=FALSE], sparse=FALSE))[,-1]
  print(paste0("Number of columns that have been one hot encoded: ", ncol(data_table2)))
  
  ## add back the numeric columns and assign back to data_table
  print("Add back numeric columns and assign to data table")
  data_table <- data_table2[,(numCols):=data_table[,numCols,with=FALSE]]
  
  print(paste0("Final dimensions of data table after pre processing: ", nrow(data_table), " by ", ncol(data_table)))
  
  ## check validity of data
  print(paste0("Number of NA's in final data table after pre processing: ", sum(sapply(data_table, is.na))))
  print(paste0("Number of numeric's in final data table after pre processing: ", sum(sapply(data_table, is.numeric))))
  if (all(sapply(data_table, function(x) all(is.finite(x))))) {
    print("All entries in final data table after pre processing are finite")
  } else {
    print("Some entries are not finite in final data table after pre processing. Please inspect final data table")
  }
  
  ## save preprocessed file as CSV
  if (!is.null(save_csv_path)) {
    print(paste0("Saving processed data to ", save_csv_path))
    fwrite(data_table, save_csv_path)
  }
  
  return(data_table)
}
