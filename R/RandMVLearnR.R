#library(reticulate)
#library(rTorch)
RandMVLearnR=function(myseed=1234L,Xdata=Xdata, Y=Y, ncomponents=NULL,num_features=NULL,
                      outcometype=NULL,kernel_param=NULL,mylambda=NULL, max_iter_nsir=NULL,
                      max_iter_PG=NULL, update_thresh_nsir=NULL,update_thresh_PG=NULL,
                      standardize_Y=FALSE,standardize_X=FALSE){
  library(reticulate)
  #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()
  reticulate::source_python(system.file("python/main_functions_probsimplex.py",
                                        package = "RandMVLearn"))

  if (!reticulate::py_available()) {
    stop("python not available")
  }

  #set defaults
  if(is.null(myseed)){
    myseed=1234L
  }

  if(is.null(ncomponents)){
    ncomponents=0L #will automatically select
  }


  # if(is.null(num_features)){
  #   n=dim(Y)[1]
  #
  #   if(n >= 1000){
  #     num_features=300L
  #   }else{
  #     #n1=floor(n/2)
  #     num_features=NULL
  #   }
  # }

  if(is.null(outcometype)){
    outcometype='continuous'
  }

  if(is.null(max_iter_nsir)){
    max_iter_nsir=500L
  }

  if(is.null(max_iter_PG)){
    max_iter_PG=500L
  }
  if(is.null(update_thresh_nsir)){
    update_thresh_nsir=10^-6
  }

  if(is.null(update_thresh_PG)){
    update_thresh_PG=10^-6
  }

  if(is.null(standardize_Y)){
    standardize_Y=FALSE
  }

  if(is.null(standardize_X)){
    standardize_X=FALSE
  }



#call algorithm to learn parameters, binary outcome
myrmvlearn=NSIRAlgorithmFISTA(Xdata, Y, myseed, ncomponents,num_features,
                              outcometype,kernel_param,mylambda, max_iter_nsir,
                              max_iter_PG, update_thresh_nsir,update_thresh_PG,
                              standardize_Y,standardize_X)

return( myrmvlearn)

}
