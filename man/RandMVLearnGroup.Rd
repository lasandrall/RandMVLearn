\name{RandMVLearnGroup}
\alias{RandMVLearnGroup}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Trains a randomized nonlinear model for simultaneous association and prediction of multiview data when there is group information for one or more views.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Trains a randomized nonlinear model for simultaneous association and prediction of multiview data. Use this function if there is prior information (group information) for at least one view.  Currently works for binary or continuous outcome.  Returns selected features, groups, model trained, view-independent low-dimensional representation(s), which could be used in subsequent analyses.
}
\usage{
RandMVLearnGroup(myseed=1234L,Xdata=Xdata, Y=Y,hasGroupInfo=GroupInfo,
                GroupIndices=groupsd,myrho=NULL,myeta=NULL,ncomponents=NULL,
                num_features=NULL,outcometype=NULL,kernel_param=NULL,
                mylambda=NULL, max_iter_nsir=NULL,max_iter_PG=NULL,
                update_thresh_nsir=NULL,update_thresh_PG=NULL,
                standardize_Y=FALSE,standardize_X=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

  \item{myseed}{An integer to set a seed. Need to append a letter L to the integer, for example 1234L. This argument can be NULL.}

  \item{Xdata}{A list of d elements, where d is the number of views for the training data. Each element is a view with dimension \eqn{n \times p^d}, where observations are on the rows and features are on the columns. The number of samples are the same across all views but \eqn{p^d} can be different.}

  \item{Y}{An \eqn{n \times q} matrix of responses. Currently allows for binary and continuous outcomes. Allows for multiple continuous outcomes, so that \eqn{q >1}. }

   \item{hasGroupInfo}{A list of d elements indicating whether or not the \eqn{dth} view has prior information. If view \eqn{d} has prior information, denote as 1, otherwise 0.   }

  \item{GroupIndices}{A list of d elements containing group information. If there is no group information for view \eqn{d}, enter NULL. Group information for view \eqn{d} is a matrix with two columns. The first column is the group number \eqn{1,2,...} and the second column is the variables in that group. Method works for non-overlapping groups.      }

\item{myrho}{ A list of \eqn{d} entries. \eqn{\rho >0} controls the amount of sparsity, for a fixed \eqn{\eta}.  }

\item{myeta}{ A list of \eqn{d} entries. \eqn{0 \le \eta \le 1} allows to select groups and variables within groups, for views with group information.  This parameter is not tuned. For a fixed \eqn{\rho}, smaller values encourage grouping (i.e. i.e. more nonzero groups are selected) and individual variable selection within groups (i.e more variables tend to have nonzero coefficients within groups); larger variables discourage group selection and encourage sparsity within group. If view \eqn{d} has no group information, set as 0. Default is 0.5 when there's group information and 0 when there's no group information.}

  \item{ncomponents}{An integer for number of low-dimensional components. Need to append a letter L to the integer. Set to 0L or NULL to allow algorithm to adaptively choose the number of components. }

  \item{num_features}{An integer for number of random mappings, typically less than the number of samples. Need to append a letter L to the integer. This argument can be NULL. If NULL, the algorithm will set it to 300 if \eqn{n \ge 1000} or \eqn{n/2} if \eqn{n < 1000}. }

  \item{outcometype}{A string for the type of outcome. Required. Either "binary" or "continuous". If not specified, will default to continuous, which might not be ideal for the type of outcome.}

  \item{kernel_param}{A list of  \eqn{d}  integers specifying the kernel parameters for the Gaussian kernel. If NULL, algorithm will choose kernel parameters for each view using median heuristic. }

  \item{mylambda}{A list of  \eqn{d}  integers specifying the regularization parameters controlling the trade-off between model fit and complexity. Default is 1 for each view. }

  \item{max_iter_nsir}{An integer indicating the number of iterations for the alternating minimization algorithm. Need to append a letter L to the integer. If NULL, defaults to 500.}

  \item{max_iter_PG}{An integer  indicating the number of iterations for the accelerated projected gradient descent algorithm for sparse learning. Need to append a letter L to the integer. If NULL, defaults to 500.}

  \item{update_thresh_nsir}{Threshold for convergence of alternating minimization algorithm. Defaults to \eqn{10^{-6}}}

  \item{update_thresh_PG}{Threshold for convergence of accelerated projected gradient descent algorithm. Defaults to \eqn{10^{-6}}}

  \item{standardize_Y}{TRUE or FALSE. If TRUE, Y will be standardized to have mean zero and variance one. Applicable to continuous outcome. Defaults to FALSE, at which point Y is centered.}

    \item{standardize_X}{TRUE or FALSE. If TRUE, each variable in each view will be standardized to have mean zero and variance one. Defaults to FALSE. }
}
\details{
%%  ~~ If necessary, more details than the description above ~~
Please refer to main paper for more details, \url{https://arxiv.org/abs/2304.04692}
}
\value{
%%  ~Describe the value returned
The function will return  a list of 17 elements. To see the elements, use double square brackets. See below for more detail of the main output.
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
Some of the arguments are needed to  proceed with testing  or prediction.
\item{Z}{A list of \eqn{d, d=1,\ldots,D} randomized nonlinear feature data for each view.   }
\item{Ghat}{ A matrix of \eqn{n \times r} joint nonlinear low-dimensional representations learned from the training data. Here, \eqn{r} is the number of latent components. This matrix could be used for further downstream analyses such as clustering.}
\item{myb}{A list of \eqn{d, d=1,\ldots,D} uniform random variables used in generating random features. }
\item{gamma}{A list with \eqn{d, d=1,\ldots,D} entries for each view. Each entry is a length-\eqn{p^d} vector of probability estimate for each variable. A value of 0 indicates the variable is not selected. }
\item{Var_selection}{A list with \eqn{d, d=1,\ldots,D} entries of variable selection for each view . Each entry is a length-\eqn{p^d} indicator vector. A value of 0 indicates the variable is not selected and a value of 1 indicates the variable is selected. }
\item{GroupSelection}{A list with \eqn{d, d=1,\ldots,D} entries of group selection for each view . Each entry contains a \eqn{G \times 2} matrix, \eqn{G} is the number of groups, the first column is the group indices and the second column is the number of variables selected in that group. If no variable is selected, we assign a zero value for that group. If there's no group information for view d, the \eqn{dth} entry is assigned a zero value.}

\item{myepsilon}{A list with \eqn{d, d=1,\ldots,D} entries for each view. Each entry contains the inverse Fourier transform for the Guassian Kernel.}
\item{Ahat}{A list with \eqn{d, d=1,\ldots,D} entries for each view. Each entry is a matrix of coeffients.}
\item{thetahat}{A \eqn{M \times q} tensor of estimated regression coefficients, where \eqn{M} is the number of random features used in training.}
\item{num_features}{An integer for number of random mappings used in training, typically less than the number of samples. }
 \item{standardize_Y}{TRUE or FALSE. If TRUE, Y was standardized to have mean zero and variance one during training of the model. Applicable to continuous outcome. If FALSE, Y was centered to have mean zero. Defualts to FALSE if NULL.}

\item{standardize_X}{TRUE or FALSE. If TRUE, each variable in each view was standardized to have mean zero and variance one when training the model. Defualts to FALSE if NULL.}
 \item{ncomponents}{An integer for number of low-dimensional components used in training. }

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
\code{\link{generateData},\link{RandMVPredict},\link{cvRandMVLearnGroup},\link{RandMVLearnR}}
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

################# train with fixed number of components and number of features
######assume all views have group information
GroupInfo=list(1,1)
g1=cbind(matrix(1,nrow=20,ncol=1),1:20) #signal variables form group 1
g2=cbind(matrix(2,nrow=1000-20,ncol=1),21:1000) #noise variables form group 2
groupsd=list(rbind(g1,g2),rbind(g1,g2))

myrho.parameter=list(0.000008, 0.000008)
RandMVGroupTrain=RandMVLearnGroup(myseed=1234L,Xdata=Xdata, Y=Y,hasGroupInfo=GroupInfo,
                          GroupIndices=groupsd,myrho=myrho.parameter,myeta=NULL,ncomponents=5L,
                          num_features =300L, outcometype=outcometype, standardize_X = FALSE)

#View GroupsSelected for Views 1 and 2 and number of variables selected in groups
GroupsSelected1=RandMVGroupTrain[["GroupSelection"]][[1]]
GroupsSelected2=RandMVGroupTrain[["GroupSelection"]][[2]]

#View Variables Selected
VarSelected1=RandMVGroupTrain[["Var_selection"]][[1]]
VarSelected2=RandMVGroupTrain[["Var_selection"]][[2]]

#####Predict an outcome using testing data assuming model has been learned
Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
Xtestdata=list(Xtest1,Xtest2)

Ytestdata=mydata[["TestData"]][[1]][["Y"]]

predictY=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
                      myEstimates=RandMVGroupTrain)
#obtain test error
test.error=predictY["TestError"]

######assume only view 1 has group information
GroupInfo=list(1,0)
g1=cbind(matrix(1,nrow=20,ncol=1),1:20) #signal variables form group 1
g2=cbind(matrix(2,nrow=1000-20,ncol=1),21:1000) #noise variables form group 2
groupsd=list(rbind(g1,g2), NULL)

myrho.parameter=list(0.000008, 0)
RandMVGroupTrain=RandMVLearnGroup(myseed=1234L,Xdata=Xdata, Y=Y,hasGroupInfo=GroupInfo,
                          GroupIndices=groupsd,myrho=myrho.parameter,myeta=NULL,ncomponents=5L,
                          num_features =300L, outcometype=outcometype, standardize_X = FALSE)


#View groups selected for View 1 and number of variables selected in groups
GroupsSelected=RandMVGroupTrain[["GroupSelection"]][[1]]

#count number of nonzero variables in each view
VarSel1=sum(RandMVGroupTrain$Var_selection[[1]]==1)
VarSel2=sum(RandMVGroupTrain$Var_selection[[2]]==1)


#####Predict an outcome using testing data assuming model has been learned
Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
Xtestdata=list(Xtest1,Xtest2)

Ytestdata=mydata[["TestData"]][[1]][["Y"]]

predictY=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
                      myEstimates=RandMVGroupTrain)
#obtain test error
test.error=predictY["TestError"]

################# train with adaptively choosen number of components and number of features
######assume all views have group information
GroupInfo=list(1,1)
g1=cbind(matrix(1,nrow=20,ncol=1),1:20) #signal variables form group 1
g2=cbind(matrix(2,nrow=1000-20,ncol=1),21:1000) #noise variables form group 2
groupsd=list(rbind(g1,g2),rbind(g1,g2))

myrho.parameter=list(0.000008, 0.000008)
RandMVGroupTrain.Adapt=RandMVLearnGroup(myseed=1234L,Xdata=Xdata, Y=Y,hasGroupInfo=GroupInfo,
                          GroupIndices=groupsd,myrho=myrho.parameter,myeta=NULL,ncomponents=NULL,
                          num_features =NULL, outcometype=outcometype, standardize_X = FALSE)

#View GroupsSelected for Views 1 and 2 and number of variables selected in groups
GroupsSelected1=RandMVGroupTrain.Adapt[["GroupSelection"]][[1]]
GroupsSelected2=RandMVGroupTrain.Adapt[["GroupSelection"]][[2]]

#View Variables Selected
VarSelected1=RandMVGroupTrain.Adapt[["Var_selection"]][[1]]
VarSelected2=RandMVGroupTrain.Adapt[["Var_selection"]][[2]]

#####Predict an outcome using testing data assuming model has been learned
Xtest1=mydata[["TestData"]][[1]][["X"]][[1]]
Xtest2=mydata[["TestData"]][[1]][["X"]][[2]]
Xtestdata=list(Xtest1,Xtest2)

Ytestdata=mydata[["TestData"]][[1]][["Y"]]

predictY=RandMVPredict(Ytest=Ytestdata,Ytrain=Y,Xtest=Xtestdata,Xtrain=Xdata,
                      myEstimates=RandMVGroupTrain.Adapt)
#obtain test error
test.error=predictY["TestError"]


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
