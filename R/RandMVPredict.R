#library(reticulate)
#library(rTorch)
RandMVPredict=function(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,outcometype=NULL,
                       myEstimates=NULL,standardize_Y=NULL,standardize_X=NULL){

  library(reticulate)
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



  mypredict=PredictYNew(Ytest,Ytrain,Xtest,Xtrain,Gtrain,myb=myb,gamma=gamma,thetahat=thetahat,
                        Ahat=Ahat,myepsilon=myepsilon,outcometype=outcometype,num_features=num_features,
                        standardize_Y=standardize_Y,standardize_X=standardize_X)

  return(mypredict)

}
