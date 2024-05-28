functions {
	// function to return log-likelihood of Gumbel copula
	real gumbel_copula(real u, real v, real theta) {
		if (theta < 1){
			reject("Theta must be greater than or equal to 1!");
		}
		real uterm = (-log(u))^theta;
		real uuterm = (-log(u))^(theta - 1);
		real vterm = (-log(v))^theta;
		real vvterm = (-log(v))^(theta - 1);
		real cterm = exp(-(uterm + vterm)^(1 / theta));
		return log(1 / (u * v) * cterm * uuterm * (-1 + theta + (uterm + vterm)^(1 / theta)) * (uterm + vterm)^(-2 + 1 / theta) * vvterm);
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
		ll[i] = gumbel_copula(u[i], v[i], theta);
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
			loglike[i] = gumbel_copula(u[i], v[i], theta);
		}
	}
}
