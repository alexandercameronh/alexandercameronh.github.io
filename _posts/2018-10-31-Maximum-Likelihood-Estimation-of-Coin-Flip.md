# Maximum Likelihood Estimation of Coin Flip

Imagine that you're being asked to flip a coin *n* times and to estimate the probability of the coin being heads-side-up. Typically the probability would be 0.5 (50%), but in this scenario we don't know if it's a fair coin.
That is, we don't know the probability, *p*.

This type of experiment is known as a Bernoulli trial. 

Where we have:
- two outcomes (heads and tails), *0* and *1* 
- probability of success, *p*
- probability of failure, *(1-p)*
- number of trials (coin flips), *n*

So we have 100 coin flips with results $x_1, x_2, ..., x_{100}$. If we were to flip the coin 100 times and we see heads 55 times, we can say that $p = 55/100 = 0.55$.
But let's prove that. Let's prove $\hat{p} = \frac{\sum x_i}{n}$


We begin with our probability mass function: $$f(x;p) = p^x(1-p)^{(1-x)}$$

All of our observations ($x_1, x_2, ..., x_n$) are independent. So the joint probability mass function is:
$$f(x_1, ..., x_n;p) = \prod_i f(x_i, p) = p^{\sum x_i}(1-p)^{(n-\sum x_i)}$$

Interpreting this as a function of the parameter ($p$), given the observations,
we get the likelihood function:
$$\mathcal{L}(p) = p^{\sum x_i}(1-p)^{(n-\sum x_i)}$$

We want to maximize this function. We want to find $p$ such that we maximize the likelihood of seeing the given observations.

[Finding a maximum](http://clas.sa.ucsb.edu/staff/lee/Max%20and%20Min's.htm) of a function typically involves finding the derivative and setting it to zero, yes?

Taking the derivative of $\mathcal{L}$ with respect to $p$ is not straightforward.
To make things easier, we take the derivative of the natural log of $\mathcal{L}$. This is a common trick -- the natural log function is a monotonically increasing function so the
value of $p$ that maximizes $ln(\mathcal{L}(p))$ also maximizes $\mathcal{L}(p)$.

$$ln(\mathcal{L}(p)) = \sum x_i ln(p) + (n-\sum x_i)ln(1-p)$$

$$\frac{\partial ln(\mathcal{L}(p))}{\partial p} = \frac{\sum x_i}{p} + \frac{(n-\sum x_i)}{(1-p)} = 0$$

$$ p(1-p)*[\frac{\sum x_i}{p} + \frac{(n-\sum x_i)}{(1-p)}] = (\sum x_i)(1-p) - (n-\sum x_i)p = 0 $$

$$ \sum x_i - p \sum x_i - np + p \sum x_i = 0 $$

Leaves us with:

$$\hat{p} = \frac{\sum_{i=1}^n x_i}{n}$$

d:)