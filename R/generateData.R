
#library(rTorch)
generateData=function(myseed=1234L,n1=500L,n2=200L,p1=1000L,p2=1000L,nContVar=1L,
                      sigmax1=0.1,sigmax2=0.1,sigmay=0.1,sigmax11=0.1,sigmax12=0.1,
                      ncomponents=3L,nReplicate=1L,outcometype='continuous'){

  # prepare python
  #generate data with two views
  library(reticulate)
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
    }else if(outcometype=='binary'){
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
  }else if(outcometype=='binary'){
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
