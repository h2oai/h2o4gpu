dyn.load("pow_wrp.so")
 mkl_pow <- function(n, x, y) .Call("mkl_vdpow", n, x, y)
 n <- 1000000
 x <- runif(n, min=2, max=10)
 y <- runif(n, min=-2, max=-1)
 start <- proc.time()
 z <- mkl_pow(n, x, y)
 end1 <- proc.time() - start
end1
##n <-1000000
i <- n
start <- proc.time()
repeat{ z[i] <- x[i]^y[i]
 i <- i - 1
 if (i==0) break() }
end2 <- proc.time() - start
end2
