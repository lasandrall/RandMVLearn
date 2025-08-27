#' Convert a List or Data Frame to a PyTorch Tensor
#'
#' This function converts an R list or data frame into a PyTorch tensor using `torch$from_numpy()`.
#'
#' @param lst A data frame, list of data frames, or numeric matrix to convert into a tensor.
#' @return A PyTorch tensor or a list of tensors.
#' @examples
#' \dontrun{
#'   torch <- reticulate::import("torch")
#'   df <- data.frame(A = rnorm(10), B = rnorm(10))
#'   tensor <- convert_to_tensor(df)
#'   print(tensor)
#' }
#' @export
convert_to_tensor <- function(lst) {
  torch <- reticulate::import("torch")

  convert_df <- function(df) {
    # Convert each column to numeric, handling factors, then reassemble as a data frame
    df_numeric <- as.data.frame(lapply(df, function(x) {
      as.numeric(if (is.factor(x)) as.character(x) else x)
    }))
    torch$from_numpy(as.matrix(df_numeric))
  }

  convert_list <- function(x) {
    if (is.data.frame(x)) {
      return(convert_df(x))
    } else if (is.matrix(x)) {
      # Convert matrix to data frame first
      return(convert_df(as.data.frame(x)))
    } else if (is.list(x)) {
      return(lapply(x, convert_list))
    } else if (is.numeric(x)) {
      # In case x is a numeric vector, convert to data frame
      return(convert_df(as.data.frame(x)))
    } else {
      return(x)
    }
  }

  convert_list(lst)
}

