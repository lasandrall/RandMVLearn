\name{generateData}
\alias{generateData}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Generate a binary or continuous multiview data having nonlinear relationships.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function is used to generate binary or continuous data for two views. Please refer to the manuscript for data generation process. Function can generate data with multiple continuous outcomes.
}
\usage{
generateData(myseed=1234L,n1=500L,n2=200L,p1=1000L,p2=1000L,nContVar=1L,
                      sigmax1=0.1,sigmax2=0.1,sigmay=0.1,sigmax11=0.1,sigmax12=0.1,
                      ncomponents=3L,nReplicate=1L,outcometype='continuous')
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{myseed}{An integer to set a seed. Need to append a letter L to the integer, for example 1234L. This argument can be NULL.}


  \item{n1}{An even integer for number of samples. If outcometype is continuous, this is the number of samples for each view. If outcometype is binary, this is the number of samples for class 1.  Need to append a letter L to the integer. Can be set to NULL. }

  \item{n2}{An even integer for number of samples in class 2 if outcome type is binary. If outcometype is continuous, this is not used.  Need to append a letter L to the integer. Can be set to NULL. }

\item{p1}{An integer for number of variables in view 1. Need to append a letter L to the integer. Can be set to NULL. }

\item{p2}{An integer for number of variables in view 2. Need to append a letter L to the integer. Can be set to NULL. For this data generation example, \eqn{p1=p2} but the method allows for different variable dimensions. }

\item{nContVar}{An integer for number of continuous outcome variables. If outcometype is binary, not used.  Need to append a letter L to the integer. Can be set to NULL. Defaults to 1.}

\item{ncomponents}{An integer for number of low-dimensional components. Need to append a letter L to the integer. Can be set to NULL. Defaults to 3.}

\item{sigmax1,sigmax2}{Variances for Views 1 and 2. Refer to manuscript for more details.}
\item{sigmay}{Variance for continuous outcome. Refer to manuscript for more details.}
\item{sigmax11}{Variance for Class 1 for binary data generation.  Refer to manuscript for more details.}
\item{sigmax12}{Variances for Class 2 for binary data generation.  Refer to manuscript for more details.}


  \item{nReplicate}{An integer for number of replicates. Need to append a letter L to the integer. Can be set to NULL. Defaults to 1.}
 \item{outcometype}{A string for the type of outcome. Required. Either "binary" or "continuous". If not specified, will default to continuous.}




}
\details{
%%  ~~ If necessary, more details than the description above ~~
Please refer to main paper for more details. Paper can be found here:
\url{https://arxiv.org/abs/2304.04692}
}
\value{
%%  ~Describe the value returned
The function will return a list with 2 entries containing training and testing data.
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
The following arguments are needed if you want to proceed with testing  or prediction.
\item{TrainData}{A list containing training Views \eqn{X} and outcome \eqn{Y}. }
\item{TestData}{ A list containing testing Views \eqn{X} and outcome \eqn{Y}.}
}

\references{
Sandra E. Safo and Han Lu (2023) Scalable Randomized Kernel Methods for Multiview Data Integration and Prediction
\url{https://arxiv.org/abs/2304.04692}
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
\code{\link{RandMVLearnR},\link{RandMVPredict}}
}

\examples{
####### generate train and test data with binary outcome- refer to manuscript for data generation

outcometype='binary'
mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,
                    sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)

#create a list of two views for training data
X1=mydata[["TrainData"]][[1]][["X"]][[1]]
X2=mydata[["TrainData"]][[1]][["X"]][[2]]
Xdata=list(X1,X2)

#training outcome
Y=mydata[["TrainData"]][[1]][["Y"]]

#testing data and outcome
Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
Xtestdata=list(Xtest1,Xtest2)

Ytestdata=mydata[["TestData"]][[1]][["Y"]]

####### generate train and test data with two continuous outcomes- refer to manuscript for data generation

outcometype='continuous'
mydata=generateData(n1=500L,n2=200L,p1=1000L,p2=1000L,sigmax11=0.1,nContVar=2L,
                    sigmax12=0.1,sigmax2=0.2,outcometype=outcometype)

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
