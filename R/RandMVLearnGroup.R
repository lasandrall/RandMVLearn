#library(reticulate)
#library(rTorch)
RandMVLearnGroup=function(myseed=1234L,Xdata=Xdata, Y=Y,hasGroupInfo=GroupInfo, GroupIndices=groupsd,
                      myrho=NULL,myeta=NULL,ncomponents=NULL,num_features=NULL,outcometype=NULL,kernel_param=NULL,mylambda=NULL,
                      max_iter_nsir=NULL,max_iter_PG=NULL, update_thresh_nsir=NULL,update_thresh_PG=NULL,
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

  #if hasGroupInfo is null, assumes there's no group information for any of the variables. Default to RandMVLearnR

  # if(is.null(myrhomin)){
  #   myrhomin=list() # a list with D entries for each view
  # }
  #
  # if(is.null(myrhomax)){
  #   myrhomax=list() # a list with D entries for each view
  # }


  if(is.null(ncomponents)){
    ncomponents=0L #will automatically select
  }



  # if(is.null(num_features)){
  #   n=dim(Y)[1]
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

  if(is.null(myeta)){
    myeta=list()
    for(d in 1:length(Xdata)){
      if(hasGroupInfo[d]==1){
        myeta[d]=0.5
      }
      else{
        myeta[d]=0.0
      }
    }
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

# #check inputs for training data
#   dsizes=lapply(Xdata, function(x) dim(x))
#   n=dsizes[[1]][1]
#   nsizes=lapply(Xdata, function(x) dim(x)[1])
#
#   if(all(nsizes!=nsizes[[1]])){
#     stop('The datasets  have different number of observations')
#   }

  # #check inputs for testing data
  # ntestsizes=lapply(Xtestdata, function(x) dim(x)[1])
  # if(all(ntestsizes!=ntestsizes[[1]])){
  #   stop('The testing datasets  have different number of observations')
  # }

  # #check data
  # if (is.list(Xdata)) {
  #   D = length(Xdata)
  #   if(D==1){
  #     stop("There should be at least two datasets")
  #   }
  # } else {
  #   stop("Input data should be a list")
  # }

#call algorithm to learn parameters, binary outcome
if(is.null(hasGroupInfo)){
            myrmvlearn=NSIRAlgorithmFISTA(Xdata, Y, myseed, ncomponents,num_features,
                              outcometype,kernel_param,mylambda, max_iter_nsir,
                              max_iter_PG, update_thresh_nsir,update_thresh_PG,
                              standardize_Y,standardize_X)
}else{
  # myrmvlearn=NSIRAlgorithmFISTASIndandGLasso(Xdata, Y,myseed, ncomponents,num_features, hasGroupInfo,
  #                                            outcometype,kernel_param,mylambda, myrho, myeta,grpweights, groups,max_iter_nsir,
  #                                            max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)
  myrmvlearn=NSIRAlgorithmFISTASIndandGLasso(Xdata, Y,myseed, ncomponents,num_features, hasGroupInfo,GroupIndices,
                                             outcometype,kernel_param,mylambda, myrho, myeta,max_iter_nsir,
                                             max_iter_PG, update_thresh_nsir,update_thresh_PG,standardize_Y,standardize_X)




}

return( myrmvlearn)

}
