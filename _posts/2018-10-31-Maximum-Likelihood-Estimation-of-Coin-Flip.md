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
$$f(x_1, ..., x_n;p) = \prod_i f(x_i, p) = p^{\sum x_i}(1-p)^{(1-\sum x_I)}$$

