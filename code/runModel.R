# load packages
library("VineCopula")
library("copula")
library("dplyr")
library("rstan")
library("parallel")
library("loo")

# number of CPU cores to use (preferably equal to the number of MCMC chains)
nc <- 4

# set number of CPU cores
Sys.setenv("MC_CORES" = nc)

# compile Stan model
lgp.model <- stan_model("dynamicFrank.stan")

# read data
data <- read.csv("Houston-Q-S.csv")

# convert discharge (Q) and surge (S) to psedo-observations
pseudo.data <-
  data.frame(u = pobs(data[["Q"]]), v = pobs(data[["S"]]))

# extract covariate
x <- data[["ONI"]]

# list data as input to Stan
feed.data <-
  list(
    N = nrow(pseudo.data),
    u = pseudo.data[["u"]],
    v = pseudo.data[["v"]],
    x = x
  )

# sample from the posterior
fit.model <-
  sampling(
    object = lgp.model,
    data = feed.data,
    chains = nc,
    cores = nc,
    iter = 4000,
    warmup = 2000,
    seed = 1999,
    control = list(adapt_delta = 0.97, max_treedepth = 12),
    sample_file = "Chain.txt",
    verbose = TRUE,
    show_messages = TRUE
  )

# get results and store them in a dataframe
results <- summary(fit.model)$summary
results <- as.data.frame(results)

# save fit Stan model
fit.model@stanmodel@dso <- new("cxxdso")
saveRDS(fit.model, file = "rstanmodel.rds")

# save post warm-up samples from all chains altogether
comb.chain <-
  extract(
    fit.model,
    inc_warmup = FALSE,
    include = TRUE,
    permuted = TRUE
  )
comb.chain <- do.call(cbind.data.frame, comb.chain)

# compute WAIC metric
loglike <- extract_log_lik(fit.model,
                           parameter_name = "loglike",
                           merge_chains = TRUE)
lplm <- waic(loglike)
print(lplm)

# compute DIC metric
logLikelihood <- function(theta) {
  b <- theta[1]
  w <- theta[2]
  ll <- 0
  for (j in 1:nrow(pseudo.data)) {
    copu <- frankCopula(param = b + w * x[j])
    ll <-
      ll + dCopula(as.matrix(pseudo.data[j, ]), copu, log = TRUE)
  }
  return (ll)
}

D <- -2 * comb.chain[["lp__"]]
D.bar <- mean(D)
D.bayes <-
  -2 * logLikelihood(c(mean(comb.chain[["b"]]), mean(comb.chain[["w"]])))

DIC <- 2 * D.bar - D.bayes
print(DIC)
