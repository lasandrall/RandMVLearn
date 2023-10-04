#library(reticulate)
#library(rTorch)
cvRandMVLearnGroup=function(myseed=1234L,Xdata=Xdata, Y=Y, hasGroupInfo=GroupInfo,
                      GroupIndices=groupsd, rhoLower=NULL,rhoUpper=NULL,myeta=NULL,
                      ncomponents=NULL,num_features=NULL,outcometype=NULL,kernel_param=NULL,mylambda=NULL,
                      numbercores=NULL,gridMethod=NULL,nfolds=3L,ngrid=8L,
                      max_iter_nsir=NULL,max_iter_PG=NULL, update_thresh_nsir=NULL,update_thresh_PG=NULL,
                      standardize_Y=FALSE,standardize_X=FALSE){

  library(reticulate)
  reticulate::source_python(system.file("python/main_functions_probsimplex.py",
                                      package = "RandMVLearn"))
    #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()

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


  # if(is.null(myeta)){
  #   myeta=list()
  #   for(d in 1:length(Xdata)){
  #     if(hasGroupInfo[d]==1){
  #       myeta[d]=0.5
  #     }
  #     else{
  #       myeta[d]=0.0
  #     }
  #   }
  # }

  # if(is.null(numbercores)){
  #   registerDoParallel()
  #   ncores=getDoParWorkers()
  #   numbercores=as.integer(ceiling(ncores/2)+1)
  # }

  if(is.null(gridMethod)){
    gridMethod='RandomSearch'
  }

  if(is.null(nfolds)){
    nfolds=3L
  }


  if(is.null(ngrid)){
    ngrid=8L
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



  #check data
  if (is.list(Xdata)) {
    D = length(Xdata)
    if(D==1){
      stop("There should be at least two datasets")
    }
  } else {
    stop("Input data should be a list")
  }

#call algorithm to learn parameters
if(is.null(hasGroupInfo)){
            myrmvlearn=NSIRAlgorithmFISTA(Xdata, Y, myseed, ncomponents,num_features,
                              outcometype,kernel_param,mylambda, max_iter_nsir,
                              max_iter_PG, update_thresh_nsir,update_thresh_PG,
                              standardize_Y,standardize_X)
            result=myrmvlearn
}else{

  # if(is.null(myeta)){
  #   myeta=list()
  #   for(d in 1:length(Xdata)){
  #     if(hasGroupInfo[d]==1){
  #       myeta[d]=0.5
  #     }
  #     else{
  #       myeta[d]=0.0
  #     }
  #   }
  # }
  library(doParallel)
  if(is.null(numbercores)){
    doParallel::registerDoParallel()
    ncores=getDoParWorkers()
    numbercores=as.integer(ceiling(ncores/2)+1)
  }
  myrmvlearn2=nsrf_cvindgrouplassoNew(Xdata, Y,myseed,ncomponents,num_features,hasGroupInfo,numbercores,
                                     outcometype,kernel_param,mylambda, myeta, rhoLower, rhoUpper,GroupIndices,
                                     gridMethod,nfolds,ngrid,max_iter_nsir, max_iter_PG, update_thresh_nsir,
                                     update_thresh_PG,standardize_Y,standardize_X)
  myrmvlearn=myrmvlearn2[['myalg']]
  result=myrmvlearn
}


return( result)

}
