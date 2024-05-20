functions {
	// function to return log-likelihood of Frank copula
	real frank_copula(real u, real v, real theta) {
		return log(theta * (1 - exp(-theta))) - theta * (u + v) -
		log( (1 - exp(-theta) - (1 - exp(-theta * u)) * (1 - exp(-theta * v)))^2);
	}
}
data {
	int<lower=0> N; // number of bivariate pairs
	vector[N] u; // pseudo-observations of first variable
	vector[N] v; // pseudo-observations of second variable
	vector[N] x; // climatic index covariate
}
parameters {
	real b; // bias
	real w; // weight (i.e., effect of climatic index on dependence)
}
model {
	vector[N] ll; // point-wise log-likelihood (pwll)
	real theta; // copula parameter
	for (i in 1:N){
		theta = b + w * x[i]; // copula parameter as a linear function of climatic index
		ll[i] = frank_copula(u[i], v[i], theta);
		target += ll[i]; // sum of pwll
	}
}
// log-likelihood for each point and MCMC iteration
generated quantities{
	vector[N] loglike;
	{
		real theta;
		for (i in 1:N){
			theta = b + w * x[i];
			loglike[i] = frank_copula(u[i], v[i], theta);
		}
	}
}
