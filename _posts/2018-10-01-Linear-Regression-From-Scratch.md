---
title:  "Linear Regression From Scratch"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


I was writing a blog post (Logistic Regression From Scratch), in which I made multiple references
 to linear regression. Understanding linear regression is somewhat beneficial when learning logsitic regression 
 so I decided to post this before posting the other.
 
 
So, linear regression.

Linear regression is a supervised learning problem begins with this:

We have our input vector $X^T = (X_1, X_2, ..., X_p)$ and from this we want to predict an output $Y$.

Since it is a regression problem, $Y$ is a real-valued output (e.g. *5.23* or *99000* and not *blue* or *success*). 

Our model will be of the form:

$$ f(X) = \beta_0 + \sum_{j=1}^p X_j\beta_j $$

or if you just want to let x_0 to always be 1 then this can be simplified to:

$$ f(X) = \sum_{j=0}^p X_j\beta_j = \beta^TX $$

