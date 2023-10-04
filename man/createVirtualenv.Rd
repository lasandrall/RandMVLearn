\name{createVirtualenv}
\alias{createVirtualenv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Create a Python virtual environment with necessary packages.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function is used to create a Python virtual environment (called "RandMVLearn_env") and installs
necessary packages such as torch, numpy, and scikit-learn to the environment. Once created, the
environment will automatically be used upon loading the package.
}
\usage{
createVirtualenv()
}
%- maybe also 'usage' for other objects documented here.
\arguments{

}
\details{
If there is an error installing the Python packages, try restarting your computer and running R/RStudio as administrator.

If the virtual environment is not created, the user's default system Python installation will be used.
In this case, the user will need to have the following packages in their main local Python installation:
  \itemize{
    \item torch
    \item matplotlib
    \item joblib
    \item scikit-learn
    \item numpy
  }

Alternatively, the user can use their own virtual environment with reticulate by activating it with
reticulate::use_virtualenv() or a similar function prior to loading RandMVLearn.

}
\value{
}

\references{
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
}

\examples{
###### create Python virtual environment "RandMVLearn_env"
createVirtualenv()

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
