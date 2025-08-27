

#' @title Generate binary or continuous nonlinear multiview data
#'
#' @description This function is used to generate binary or continuous nonlinear data for two views.
#' Please refer to the manuscript for data generation process.
#' Function can generate data with multiple continuous outcomes.
#'
#' @param myseed An integer to set a seed. Need to append a letter L to the integer, for example 1234L.
#' This argument can be NULL.
#' @param n1 An even integer for number of samples. If outcometype is continuous,
#' this is the number of samples for each view. If outcometype is categorical,
#' this is the number of samples for class 1.  Need to append a letter L to the integer.
#' Can be set to NULL.
#' @param n2 An even integer for number of samples in class 2 if outcome type is categorical.
#' If outcometype is continuous, this is not used.  Need to append a letter L to the integer.
#' Can be set to NULL.
#' @param p1 An integer for number of variables in view 1. Need to append a letter L to the integer.
#' Can be set to NULL.
#' @param p2 An integer for number of variables in view 2. Need to append a letter L to the integer.
#' Can be set to NULL. For this data generation example, \eqn{p1=p2} but the method allows
#' for different variable dimensions.
#' @param nContVar An integer for number of continuous outcome variables. If outcometype is categorical, not used.
#' Need to append a letter L to the integer. Can be set to NULL. Defaults to 1.
#' @param sigmax1 Variance for View 1. Refer to manuscript for more details.
#' @param sigmax2 Variance for View 2. Refer to manuscript for more details.
#' @param sigmay Variance for continuous outcome. Refer to manuscript for more details.
#' @param sigmax11 Variance for Class 1 for binary data generation.  Refer to manuscript for more details.
#' @param sigmax12 Variances for Class 2 for binary data generation.  Refer to manuscript for more details.
#' @param ncomponents An integer for number of low-dimensional components. Need to append a letter L to the integer.
#' Can be set to NULL. Defaults to 3.
#' @param nReplicate An integer for number of replicates. Need to append a letter L to the integer.
#' Can be set to NULL. Defaults to 1.
#' @param outcometype A string for the type of outcome. Required. Either "categorical" or "continuous".
#' If not specified, will default to continuous.
#'
#' @details Please refer to main paper for more details. Paper can be found here:
#' \url{https://arxiv.org/abs/2304.04692}
#'
#' @return The function will return a list with 2 entries containing training and testing data.
#' The following arguments are needed if you want to proceed with testing  or prediction.
#'  \item{TrainData}{A list containing training Views \eqn{X} and outcome \eqn{Y}.}
#'  \item{TestData}{A list containing testing Views \eqn{X} and outcome \eqn{Y}.}
#'
#'
#' @seealso \code{\link{RandMVLearnR}}  \code{\link{RandMVPredict}}
#'
#' @author
#' Sandra E. Safo
#'
#' @references
#' Sandra E. Safo and Han Lu (2024) Scalable Randomized Kernel Methods for Multiview Data Integration and Prediction
#' Accepted in Biostatistics. \url{https://arxiv.org/abs/2304.04692}
#'
#' @import reticulate
#' @importFrom reticulate source_python py_config py_available
#'
#' @export
#' @examples
#'  ####### generate train and test data with binary outcome- refer to manuscript for data generation
#' createVirtualenv()
#'
#' outcometype='categorical'
#' mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,
#'                    sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)
#'
#' #create a list of two views for training data
#' X1=mydata[["TrainData"]][[1]][["X"]][[1]]
#' X2=mydata[["TrainData"]][[1]][["X"]][[2]]
#' Xdata=list(X1,X2)
#'
#' #training outcome
#' Y=mydata[["TrainData"]][[1]][["Y"]]
#'
#' #testing data and outcome
#' Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
#' Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
#' Xtestdata=list(Xtest1,Xtest2)
#'
#' Ytestdata=mydata[["TestData"]][[1]][["Y"]]
#'
#' ####### generate train and test data with two continuous outcomes- refer to manuscript for data generation
#'
#' outcometype='continuous'
#' mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,nContVar=2L,
#'                    sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)
#'
generateData=function(myseed=1234L,n1=500L,n2=200L,p1=1000L,p2=1000L,nContVar=1L,
                      sigmax1=0.1,sigmax2=0.1,sigmay=0.1,sigmax11=0.1,sigmax12=0.1,
                      ncomponents=3L,nReplicate=1L,outcometype='continuous'){

  # prepare python
  #generate data with two views
  # library(reticulate)
  reticulate::source_python(system.file("python/main_functions_probsimplex.py",
                                        package = "RandMVLearn"))

  #reticulate::use_python(python = Sys.which("python"))

  reticulate::py_config()
  print(reticulate::py_config())

  if (!reticulate::py_available()){
    stop("python not available")
  }

  #set defaults
  if(is.null(myseed)){
    myseed=1234L
  }

  if(is.null(n1)){
    n1=500L
  }

  if(is.null(n2)){
    n2=200L
  }

  if(is.null(p1)){
    p1=1000L
  }

  if(is.null(p2)){
    p1=1000L
  }
  if(is.null(nContVar)){
    nContVar=1L
  }

  if(is.null(sigmax1)){
    sigmax1=0.1
  }

  if(is.null(sigmax11)){
    sigmax11=0.1
  }

  if(is.null(sigmax12)){
    sigmax12=0.1
  }

  if(is.null(outcometype)){
    outcometype='continuous'
  }

  if(is.null(sigmax2)){
    if(outcometype=='continuous'){
      sigmax2=0.1
    }else if(outcometype=='categorical'){
      sigmax2=0.2
    }
  }

  if(is.null(sigmay)){
    sigmay=0.1
  }


  if(is.null(ncomponents)){
    ncomponents=3L
  }

  if(is.null(nReplicate)){
    nReplicate=1L
  }



  TrainData=list()
  TestData=list()
  if(outcometype=='continuous'){
    for(j in 1:nReplicate){
      myseed=j+1234
      TrainData[[j]]=generateContData(myseed,n1,n2,p1,p2,nContVar,
                                    sigmax1,sigmax2,sigmay,ncomponents)
      myseed=j+12345
      TestData[[j]]=generateContData(myseed,n1,n2,p1,p2,nContVar,
                                     sigmax1,sigmax2,sigmay,ncomponents)

    }
  }else if(outcometype=='categorical'){
    for(j in 1:nReplicate){
      myseed=j+1234
      TrainData[[j]]=generateBinaryData(myseed,n1,n2,p1,p2,
                                        sigmax11,sigmax12,sigmax2)
      myseed=j+12345
      TestData[[j]]=generateBinaryData(myseed,n1,n2,p1,p2,
                                     sigmax11,sigmax12,sigmax2)

    }
  }


  result=list(TrainData=TrainData, TestData=TestData)
  return( result)

}
