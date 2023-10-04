### Loading the Python Module

# Load the module and create dummy objects from it, all of which are NULL
#the_py_module <- reticulate::import_from_path(
#  "main_functions_probsimplex",
#  file.path("inst", "python"),
#  delay_load = TRUE
#)
#for (obj in names(the_py_module)) {
#  assign(obj, NULL)
#}

# Clean up
#rm(the_py_module)

# Now all those names are in the namespace, and ready to be replaced on load
.onAttach <- function(libname, pkgname) {

  # Load the virtual environment if it exists (created with createVirtualenv() function)
  if(reticulate::virtualenv_exists(envname="RandMVLearn_env")){
    reticulate::use_virtualenv(virtualenv = "RandMVLearn_env", required=FALSE)
  } else{
    warning("It's recommended you create the package virtual environment with the createVirtualenv() function.")
  }
#
#   the_py_module <- reticulate::import_from_path(
#     "main_functions_probsimplex",
#     system.file("python", package = packageName()),
#     delay_load=TRUE
#   )
#   # assignInMyNamespace(...) is meant for namespace manipulation
#   for (obj in names(the_py_module)) {
#     assignInMyNamespace(obj, the_py_module[[obj]])
#   }
}
