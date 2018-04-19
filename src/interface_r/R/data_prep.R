#'Prep data for h2o4gpu algorithms
#'
#'@param data_table `data.table` object containing data that needs to be preprocessed for h2o4gpu
#'@param data_path Path to data that needs to be preprocessed for h2o4gpu. Only needed if `data_table` parameter is not provided/available
#'@param response Response column as a string or index
#'@param save_as_cv Whether to save processed data as a csv
#'@param output_csv_path Path to save processed data as a csv
#'@export
prep_data <- function(data_table=NULL, data_path=NULL, response=NULL, save_as_csv=FALSE, save_csv_path=NULL){
  
  if (save_as_csv) {
    if (!is.null(save_csv_path)) {
      print(paste0("CSV will be saved to -> ", save_csv_path))
    } else {
      save_csv_path <- getwd()
      print(paste0("CSV will be saved to current working directory -> ", save_csv_path, " since `save_csv_path` was not specified"))
    }
  }
  
  if (is.null(data_table)) {
    if (!is.null(data_path)) {
      print(paste0("Reading in -> ",data_path))
      DT <- fread(data_path)
    } else {
      stop("Parameters `data_table` & `data_path` are NULL. Need to specify at least one.")
    }
  } else {
    DT <- data_table
  }
  
  if (!is.null(response)) {
    response <- response
    if (is.character(response)) {
      print(paste0("Response is -> ",response))
    } else {
      print(paste0("Response is -> ",colnames(DT)[response]))
    }
  } else {
    stop("Response is not specified")
  }
  
  print(paste0("Number of columns: ", ncol(DT)))
  
  print(paste0("Number of rows: ", nrow(DT)))

  ## Label-encoding of categoricals (those cols with fewer than 1k levels, but not constant)
  print("Label encoding dataset...")
  feature.names <- setdiff(names(DT), response)
  for (ff in feature.names) {
    tt <- uniqueN(DT[[ff]])
    if (tt < 1000 && tt > 1) {
      DT[, (ff):=factor(DT[[ff]])]  
      print(paste0(ff," has ",tt," levels"))
    }
    if (tt < 2) {
      print(paste0("Dropping constant column: ", ff))
      DT[, (ff):=NULL]
    }
  }
  
  print(paste0("Number of columns after label encoding: ", ncol(DT)))
  
  numCols <- names(DT)[which(sapply(DT, is.numeric))]
  catCols <- names(DT)[which(sapply(DT, is.factor))]
  print(paste0("Number of numeric columns: ",     length(numCols)))
  print(paste0("Number of categorical columns: ", length(catCols)))
  
  ## impute missing values, drop near-const cols and standardize the data
  print("Imputing missing values using mean...")
  cols <- setdiff(numCols,c(response))
  for (c in cols) {
    DT[!is.finite(DT[[c]]), (c):=mean(DT[[c]], na.rm=TRUE)]
    if (!is.finite(sd(DT[[c]])) || sd(DT[[c]])<1e-4) 
      DT[,(c):=NULL]
    else
      DT[,(c):=scale(as.numeric(DT[[c]]))]
  }
  print(paste0("Number of columns after mean imputation: ", ncol(DT)))
  
  ## one-hot encode the categoricals
  print("One hot encoding data table categoricals only...")
  DT2 <- as.data.table(model.matrix(DT[[response]]~., data = DT[,c(catCols), with=FALSE], sparse=FALSE))[,-1]
  print(paste0("Number of columns that have been one hot encoded: ", ncol(DT2)))
  
  ## add back the numeric columns and assign back to DT
  print("Add back numeric columns and assign to data table")
  DT <- DT2[,(numCols):=DT[,numCols,with=FALSE]]
  
  print(paste0("Final dimensions of data table after pre processing: ", nrow(DT), " by ", ncol(DT)))
  
  ## check validity of data
  print(paste0("Number of NA's in final data table after pre processing: ", sum(sapply(DT, is.na))))
  print(paste0("Number of numeric's in final data table after pre processing ", sum(sapply(DT, is.numeric))))
  if (all(sapply(DT, function(x) all(is.finite(x))))) {
    print("All entries in final data table after pre processing are finite")
  } else {
    print("Some entries are not finite in final data table after pre processing. Please inspect final data table")
  }
  
  ## save preprocessed file as CSV
  if (save_as_csv) {
    print(paste0("Saving processed data to ", save_csv_path))
    fwrite(DT, output_csv_name)
  }
  
  return(DT)
}
