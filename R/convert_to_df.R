#' Convert a Tensor to a Data Frame
#'
#' This function converts a PyTorch tensor (or a list of tensors) into an R data frame.
#' If the tensor has at least two columns, the last column is assumed to be a class label.
#'
#' @param item A PyTorch tensor, a list of tensors, or another R object.
#' @return A data frame or a list of data frames.
#' @examples
#' \dontrun{
#'   torch <- reticulate::import("torch")
#'   tensor <- torch$randn(c(10, 3))  # Random 10x3 tensor
#'   df <- convert_to_df(tensor)
#'   print(df)
#' }
#' @export
convert_to_df <- function(item) {
  # Internal function to convert a 2D tensor to a data frame
  convert_tensor_to_df <- function(tensor) {
    # Convert the tensor to a matrix and then to a data frame
    df <- as.data.frame(as.matrix(tensor$numpy()))

    # If there are at least 2 columns, assume the last one is the class
    if (ncol(df) >= 2) {
      names(df) <- c(paste0("X", 1:(ncol(df) - 1)), "Class")
      df$Class <- factor(df$Class)  # Ensure the class column is a factor
    } else {
      names(df) <- paste0("X", 1:ncol(df))
    }

    return(df)
  }

  # Recursive function to handle tensors within lists
  if (is.list(item)) {
    return(lapply(item, convert_to_df))
  }

  # Check if the item is a tensor (assuming it's stored as an environment)
  if (typeof(item) == "environment") {
    return(convert_tensor_to_df(item))
  }

  return(item)
}
