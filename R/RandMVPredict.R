

#' @title Predicts a test joint nonlinear low-dimensional embedding and a test outcome.
#'
#' @description This function predicts a joint low-dimensional nonlinear embedding from test data using a learned model.
#' It also predicts a test outcome.  Use this function if there is a learned model.
#' If no learned model, the algorithm will first train a RandMVLearn model.
#' Note that for continuous outcomes, because Y is standardized
#' or centered, the predicted Y is not in the scale of the original outcome so
#' MSEs are not in the scale of original outcome. Can use Y_mean and Y_std
#' to transform predicted Y to original scale, which can be used to calculate MSE
#' in the scale of the original data.
#' @param Ytest An \eqn{ntest \times q} matrix of responses. Currently allows for categorical and continuous outcomes.
#' Allows for multiple continuous outcomes, so that \eqn{q >1}. This will be compared with the predicted Ytest.
#' @param Ytrain An \eqn{n \times q} matrix of responses for training. Currently allows for
#' categorical and continuous outcomes. Allows for multiple continuous outcomes, so that \eqn{q >1}.
#' @param Xtest A list of d elements, where d is the number of views for the testing data.
#' Each element is a view with dimension \eqn{ntest \times p^d}, where observations
#' are on the rows and features are on the columns. The number of samples are the
#' same across all views but \eqn{p^d} can be different..
#' @param Xtrain A list of d elements, where d is the number of views for the training data.
#' Each element is a view with dimension \eqn{n \times p^d}, where observations are on the
#' rows and features are on the columns. The number of samples are the same across
#' all views but \eqn{p^d} can be different.
#' @param outcometype A string for the type of outcome. Required if myEstimates is NULL.
#' Either "categorical" or "continuous". If not specified, will default to continuous,
#' which might not be ideal for the type of outcome.
#' @param myEstimates A trained RandMVLearn model. Can be NULL. If NULL, algorithm
#' will first train a RandMVLearn model.
#' @param standardize_Y TRUE or FALSE. If TRUE, Y will be standardized to have mean zero
#' and variance one during training of the model. Applicable to continuous outcome.
#' If FALSE, Y will be centered to have mean zero. Defualts to FALSE if NULL.
#' @param standardize_X TRUE or FALSE. If TRUE, each variable in each training view will
#' be standardized to have mean zero and variance one when training the model.
#' Testing data for each view will be standardized with the mean and variance
#' from training data. Defaults to FALSE if NULL.
#'
#' @details Please refer to main paper for more details. Paper can be found here:
#' \url{https://arxiv.org/abs/2304.04692}
#'
#' @return The function will return a list of 4 elements. To see the elements,
#' use double square brackets. See below for more detail of the main output.
#' The following arguments are needed if you want to proceed with testing  or prediction.
#' \item{predictedEstimates}{A list with 4 entries of prediction estimates. "PredictedEstimates$predictedYtest" is the predicted Ytest. "PredictedEstimates$predictedYtest" is the predicted Y. "PredictedEstimates$EstErrorTrain" is the  estimated train error. "PredictedEstimates$EstErrorTest" is the  estimated test error }
#' \item{TrainError}{Estimated training error}
#' \item{TestError}{Estimated testing error}
#' \item{Gtrain}{ A matrix of \eqn{n \times r} joint nonlinear low-dimensional representations predicted from the training data. Here, \eqn{r} is the number of latent components. This matrix could be used for further downstream analyses such as clustering. This matrix is used together with Gtest to predict a test outcome. }
#' \item{Gtest}{ A matrix of \eqn{ntest \times r} predicted test joint nonlinear low-dimensional representations. Here, \eqn{r} is the number of latent components. This matrix is used together with Gtrain to predict a test outcome.}
#' \item{Xtest_standardized}{A list of d elements, where d is the number of views for the testing data used for prediction. Each element is a view with dimension \eqn{ntest \times p^d}, where observations are on the rows and features are on the columns. This matrix coincides with Xtest if standardize_X is FALSE or NULL.}
#'
#'
#' @seealso \code{\link{generateData},\link{RandMVLearnR},\link{RandMVLearnGroup},\link{cvRandMVLearnGroup}}
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
#' # generate train and test data with binary outcome- refer to manuscript for data generation
#' createVirtualenv()
#' outcometype='categorical'
#' mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,
#'                   sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)
#' #create a list of two views
#' X1=mydata[["TrainData"]][[1]][["X"]][[1]]
#' X2=mydata[["TrainData"]][[1]][["X"]][[2]]
#' Xdata=list(X1,X2)
#'
#' Y=mydata[["TrainData"]][[1]][["Y"]]
#'
#' ##### train with fixed number of components and number of features
#' RandMVTrain=RandMVLearnR(myseed=1234L,Xdata=Xdata, Y=Y, ncomponents=5L,num_features=300L,
#'                         outcometype=outcometype, standardize_X = FALSE)
#' #####Predict an outcome using testing data assuming model has been learned
#' Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
#' Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
#' Xtestdata=list(Xtest1,Xtest2)
#'
#' Ytestdata=mydata[["TestData"]][[1]][["Y"]]
#'
#' predictY=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
#'                       myEstimates=RandMVTrain)
#' #obtain test error
#' test.error=predictY[["Test Error"]]
#'
#' #####Predict an outcome using testing data assuming model has not been learned.
#' ###Algorithm will first train the model.
#' predictY2=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
#'                        outcometype='categorical',myEstimates=NULL)
#'
#' #obtain test error
#' test.error=predictY2[["Test Error"]]
#'
RandMVPredict=function(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,outcometype=NULL,
                       myEstimates=NULL,standardize_Y=NULL,standardize_X=NULL){

  # library(reticulate)
  reticulate::source_python(system.file("python/main_functions_probsimplex.py",
                                        package = "RandMVLearn"))

  #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()

  if (!reticulate::py_available()) {
    stop("python not available")
  }


  if(is.null(outcometype)){
    outcometype=='continuous'
  }

  if(is.null(standardize_Y)){
    standardize_Y=FALSE
  }

  if(is.null(standardize_X)){
    standardize_X=FALSE
  }



  #if not learned, call RandMVLearn
  if(is.null(myEstimates)){
    RandMVLearn=RandMVLearnR(myseed=1234L,Xdata=Xtrain, Y=Ytrain,outcometype=outcometype,
                             standardize_Y=standardize_Y,standardize_X=standardize_Y)
  }else{
    RandMVLearn=myEstimates
  }

  #set parameters
  Gtrain=RandMVLearn$Ghat
  myb=RandMVLearn$myb
  gamma=RandMVLearn$gamma
  thetahat=RandMVLearn$thetahat
  Ahat=RandMVLearn$Ahat
  myepsilon=RandMVLearn$myepsilon
  num_features=RandMVLearn$num_features
  standardize_Y=RandMVLearn$standardize_Y
  standardize_X=RandMVLearn$standardize_Y
  outcometype=RandMVLearn$outcometype
  X_mean=RandMVLearn$X_mean
  X_std=RandMVLearn$X_std
  Y_mean=RandMVLearn$Y_mean
  Y_std=RandMVLearn$Y_std
  omegaweight=RandMVLearn$mykappaweight



  mypredict=PredictYNew(Ytest,Ytrain,Xtest,Xtrain,Gtrain,myb=myb,gamma=gamma,thetahat=thetahat,
                        Ahat=Ahat,myepsilon=myepsilon,outcometype=outcometype,num_features=num_features,
                        standardize_Y=standardize_Y,standardize_X=standardize_X,
                        X_mean=X_mean,
                        X_std=X_std,
                        Y_mean=Y_mean,
                        Y_std=Y_std,
                        myomegaweight=omegaweight)

  return(mypredict)

}
