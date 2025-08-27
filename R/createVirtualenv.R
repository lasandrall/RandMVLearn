

#' @title Create a Python virtual environment with necessary packages.
#'
#' @description This function is used to create a Python virtual environment (called "RandMVLearn_env") and installs
#' necessary packages such as torch, numpy, and scikit-learn to the environment. Once created, the
#' environment will automatically be used upon loading the package.
#'
#' @details If there is an error installing the Python packages, try restarting your computer and running R/RStudio as administrator.
#'
#' If the virtual environment is not created, the user's default system Python installation will be used.
#' In this case, the user will need to have the following packages in their main local Python installation:
#'  \itemize{
#'    \item torch
#'    \item matplotlib
#'    \item joblib
#'    \item scikit-learn
#'    \item numpy
#'  }
#'
#' Alternatively, the user can use their own virtual environment with reticulate by activating it with
#' reticulate::use_virtualenv() or a similar function prior to loading RandMVLearn.
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
#' Leif Verace
#'
#' @import reticulate
#' @importFrom reticulate virtualenv_exists virtualenv_remove virtualenv_create
#'
#' @export
#' @examples
#' ###### create Python virtual environment "RandMVLearn_env"
#' createVirtualenv()
#'

createVirtualenv=function(){

  env_name="RandMVLearn_env4"
  # new_env = identical(env_name, "RandMVLearn_env")

  # if(new_env && reticulate::virtualenv_exists(envname=env_name) == TRUE){
  #   reticulate::virtualenv_remove(env_name)
  # }
  #
  # package_req <- c("torch", "matplotlib", "joblib", "scikit-learn", "numpy")
  #
  # reticulate::virtualenv_create(env_name, packages = package_req)
  if (reticulate::virtualenv_exists(env_name)) {
    # Do nothing
  } else {
    package_req <- c("torch", "matplotlib", "joblib", "scikit-learn", "numpy")
    reticulate::virtualenv_create(env_name, packages = package_req)
  }
  reticulate::use_virtualenv(env_name, required = TRUE)

  { cat(paste0("\033[0;", 32, "m","Virtual environment installed, please restart your R session and reload the package.","\033[0m","\n"))}
}
