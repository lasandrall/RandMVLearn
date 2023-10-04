createVirtualenv=function(){

  env_name="RandMVLearn_env"
  new_env = identical(env_name, "RandMVLearn_env")

  if(new_env && reticulate::virtualenv_exists(envname=env_name) == TRUE){
    reticulate::virtualenv_remove(env_name)
  }

  package_req <- c("torch", "matplotlib", "joblib", "scikit-learn", "numpy")

  reticulate::virtualenv_create(env_name, packages = package_req)

  { cat(paste0("\033[0;", 32, "m","Virtual environment installed, please restart your R session and reload the package.","\033[0m","\n"))}
}
