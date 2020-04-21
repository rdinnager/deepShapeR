library(keras)
library(tfdatasets)
library(tfautograph)
library(tensorflow)

LatentAdd <- R6::R6Class("LatentAdd",
                         
                         inherit = KerasLayer,
                         
                         public = list(
                           
                           latent_size = NULL,
                           
                           n_samps = NULL,
                           
                           initialize_sd = NULL,
                           
                           latent_codes = NULL,
                           
                           initialize = function(latent_size, n_samps, initialize_sd) {
                             self$latent_size <- latent_size
                             self$n_samps <- n_samps
                             self$initialize_sd <- initialize_sd 
                           },
                           
                           build = function(input_shape) {
                             self$latent_codes <- self$add_weight(
                               name = 'latent_codes', 
                               shape = list(self$n_samps, self$latent_size),
                               initializer = initializer_random_normal(stddev = self$initialize_sd),
                               trainable = TRUE
                             )
                           },
                           
                           call = function(x, mask = NULL) {
                             batch_latent_codes <- tf$gather(self$latent_codes, tf$squeeze(x[[2]] - 1L, 0L))
                             tf$concat(list(x[[1]], batch_latent_codes))
                           },
                           
                           compute_output_shape = function(input_shape) {
                             list(input_shape[[1]], self$latent_size)
                           },
                           
                           get_config = function() {
                             list(latent_size = self$latent_size,
                                  n_samps = self$n_samps,
                                  initialize_sd = self$initialize_sd)
                           }
                         )
)

layer_add_latent_codes <- function(object, latent_size, n_samps, initialize_sd = 0.0001, name = NULL, trainable = TRUE) {
  create_layer(LatentAdd, object, list(
    latent_size = as.integer(latent_size),
    n_samps = as.integer(n_samps),
    initialize_sd = initialize_sd,
    name = name,
    trainable = trainable
  ))
}

create_igr_net <- function(coord_dim = 2L, sdf_breadth = 512L, latent_code_size = 256L,
                    n_samps, use_mirror = TRUE, n_layers = 8L,
                    add_coords_every = 4L, 
                    use_batchnorm = FALSE,
                    activation = c("softplus", "relu"),
                    initialize_sd = 0.0001,
                    final_activation = "linear",
                    geo_init_r = 1.0,
                    add_latent = TRUE,
                    calc_grad = TRUE) {
  
  activation <- match.arg(activation)
  
  keras_model_custom(name = "ImpGeoReg", function(self) {
    
    if(add_latent) {
      
      self$latent <- layer_add_latent_codes(list(input_coords, input_ids),
                                            latent_size = latent_code_size, n_samps = n_samps,
                                            initialize_sd = initialize_sd,
                                            name = "add_latent_codes")
      
    }
    
    self$dense_layers <- list()
    
    for(i in seq_len(n_layers - 1L)) {
      
      if(i %% add_coords_every == 0) {
        
        keras::layer_dense(units = sdf_breadth, activation = activation,
                           name = paste0("layer_dense_", .x),
                           kernel_initializer = 
                             keras::initializer_random_normal(stddev = stddev = sqrt(2) / 
                                                                sqrt(sdf_breadth -
                                                                       latent_coord_len))
        )
        
      } else {
      
        keras::layer_dense(units = sdf_breadth, activation = activation,
                           name = paste0("layer_dense_", .x),
                           kernel_initializer = 
                             keras::initializer_random_normal(stddev = sqrt(2) / 
                                                                sqrt(sdf_breadth))
        )
      }
      
    }
    
                   
    self$dense_layers <- c(self$dense_layers, list(keras::layer_dense(units = sdf_breadth, activation = activation,
                                            name = paste0("layer_dense_", n_layer),
                                            kernel_initializer = keras::initializer_random_normal(mean = sqrt(pi) / 
                                                                                                    sqrt(sdf_breadth),
                                                                                                  stddev = 0.000001),
                                            bias_initializer = keras::initializer_constant(-geo_init_r))
                                            )
                           )
    
    if(use_batchnorm) {
      self$batchnorm_layers <- purrr::map(1:n_layers, 
                                          ~keras::layer_batch_normalization(name = 
                                                                              paste0("layer_bn_", .x)))
    }
    
    
    self$dense1 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense2 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense3 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense4 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense5 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense6 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense7 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    self$dense8 <- keras::layer_dense(units = sdf_breadth, activation = activation)
    #self$dense9 <- keras::layer_dense(units = sdf_breadth, activation = "relu")
    #self$dense10 <- keras::layer_dense(units = sdf_breadth, activation = "relu")
    #self$dense11 <- keras::layer_dense(units = sdf_breadth, activation = "relu")
    self$dense12 <- keras::layer_dense(units = 1, activation = "tanh")
    
    function(inputs, mask = NULL) {
      
      x <- keras::layer_concatenate(inputs) %>% 
        self$dense_start() 
      
      y1 <- x
      
      x <- x %>% 
        self$dense1() %>%
        self$dense2()
      
      x <- keras::layer_add(list(x, y1))
      
      y2 <- x
      
      x <- x %>%
        self$dense3() %>%
        self$dense4()
      
      x <- keras::layer_add(list(x, y2))
      
      y3 <- x
      
      x <- keras::layer_concatenate(list(x, inputs[[1]]))
      
      x <- x %>%
        self$dense5() %>%
        self$dense6() 
      
      x <- keras::layer_add(list(x, y3))
      
      y4 <- x
      
      x <- x %>%
        self$dense7() %>%
        self$dense8()
      
      x <- keras::layer_add(list(x, y4))
      
      # y5 <- x
      # 
      # x <- keras::layer_concatenate(list(x, inputs[[1]]))
      # 
      # x <- x %>%
      #   self$dense9() %>%
      #   self$dense10() 
      # 
      # x <- keras::layer_add(list(x, y5))
      
      x <- x %>%
        #self$dense11() %>%
        self$dense12() 
      
      x
      
    }
  })
}