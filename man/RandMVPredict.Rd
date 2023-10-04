\name{RandMVPredict}
\alias{RandMVPredict}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Predicts a test joint nonlinear low-dimensional embedding and a test outcome.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function predicts a joint low-dimensional nonlinear embedding from test data using a learned model. It also predicts a test outcome.  Use this function if there is a learned model. If no learned model, the algorithm will first train a RandMVLearn model.
}
\usage{
RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,outcometype=NULL,
              myEstimates=NULL,standardize_Y=NULL,standardize_X=NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
 \item{Ytest}{An \eqn{ntest \times q} matrix of responses. Currently allows for binary and continuous outcomes. Allows for multiple continuous outcomes, so that \eqn{q >1}. This will be compared with the predicted Ytest.  }

 \item{Ytrain}{An \eqn{n \times q} matrix of responses for training. Currently allows for binary and continuous outcomes. Allows for multiple continuous outcomes, so that \eqn{q >1}. }


 \item{Xtest}{A list of d elements, where d is the number of views for the testing data. Each element is a view with dimension \eqn{ntest \times p^d}, where observations are on the rows and features are on the columns. The number of samples are the same across all views but \eqn{p^d} can be different. If }

  \item{Xtrain}{A list of d elements, where d is the number of views for the training data. Each element is a view with dimension \eqn{n \times p^d}, where observations are on the rows and features are on the columns. The number of samples are the same across all views but \eqn{p^d} can be different.}

 \item{outcometype}{A string for the type of outcome. Required if myEstimates is NULL. Either "binary" or "continuous". If not specified, will default to continuous, which might not be ideal for the type of outcome. }


 \item{myEstimates}{A trained RandMVLearn model. Can be NULL. If NULL, algorithm will first train a RandMVLearn model.}
  \item{standardize_Y}{TRUE or FALSE. If TRUE, Y will be standardized to have mean zero and variance one during training of the model. Applicable to continuous outcome. If FALSE, Y will be centered to have mean zero. Defualts to FALSE if NULL.}

\item{standardize_X}{TRUE or FALSE. If TRUE, each variable in each training view will be standardized to have mean zero and variance one when training the model.Testing data for each view will be standardized with the mean and variance from training data. Defaults to FALSE if NULL.}


}
\details{
%%  ~~ If necessary, more details than the description above ~~
Please refer to main paper for more details.
}
\value{
%%  ~Describe the value returned
The function will return a list of 4 elements. To see the elements, use double square brackets. See below for more detail of the main output.
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
The following arguments are needed if you want to proceed with testing  or prediction.
\item{predictedEstimates}{A list with 4 entries of prediction estimates. "PredictedEstimates$predictedYtest" is the predicted Ytest. "PredictedEstimates$predictedYtest" is the predicted Y. "PredictedEstimates$EstErrorTrain" is the  estimated train error. "PredictedEstimates$EstErrorTest" is the  estimated test error }
\item{TrainError}{Estimated training error}
\item{TestError}{Estimated testing error}
\item{Gtrain}{ A matrix of \eqn{n \times r} joint nonlinear low-dimensional representations predicted from the training data. Here, \eqn{r} is the number of latent components. This matrix could be used for further downstream analyses such as clustering. This matrix is used together with Gtest to predict a test outcome. }
\item{Gtest}{ A matrix of \eqn{ntest \times r} predicted test joint nonlinear low-dimensional representations. Here, \eqn{r} is the number of latent components. This matrix is used together with Gtrain to predict a test outcome.}
 \item{Xtest_standardized}{A list of d elements, where d is the number of views for the testing data used for prediction. Each element is a view with dimension \eqn{ntest \times p^d}, where observations are on the rows and features are on the columns. This matrix coincides with Xtest if standardize_X is FALSE or NULL.}
}

\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{generateData},\link{RandMVLearnR},\link{RandMVLearnGroup},\link{cvRandMVLearnGroup}}
}

\examples{
# generate train and test data with binary outcome- refer to manuscript for data generation

outcometype='binary'
mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,
                    sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)

#create a list of two views
X1=mydata[["TrainData"]][[1]][["X"]][[1]]
X2=mydata[["TrainData"]][[1]][["X"]][[2]]
Xdata=list(X1,X2)

Y=mydata[["TrainData"]][[1]][["Y"]]

##### train with fixed number of components and number of features
RandMVTrain=RandMVLearnR(myseed=1234L,Xdata=Xdata, Y=Y, ncomponents=5L,num_features=300L,
                         outcometype=outcometype, standardize_X = FALSE)


#####Predict an outcome using testing data assuming model has been learned
Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
Xtestdata=list(Xtest1,Xtest2)

Ytestdata=mydata[["TestData"]][[1]][["Y"]]

predictY=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
                       myEstimates=RandMVTrain)

#obtain test error
test.error=predictY[["TestError"]]


#####Predict an outcome using testing data assuming model has not been learned.
###Algorithm will first train the model.
predictY2=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
                        outcometype='binary',myEstimates=NULL)

#obtain test error
test.error=predictY2[["TestError"]]


% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
}
