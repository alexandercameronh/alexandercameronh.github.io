---
title:  "Linear Regression From Scratch"
---


I was writing a blog post (Logistic Regression From Scratch), in which I made multiple references
 to linear regression. Understanding linear regression is somewhat beneficial when learning logsitic regression 
 so I decided to post this before posting the other.
 
 
So, linear regression.

Linear regression is a supervised learning problem and begins with this:

We have our input vector $$X^T = (X_1, X_2, ..., X_p)$$ and from this we want to predict an output $Y$.

Since it is a regression problem, $Y$ is a real-valued output (e.g. *5.23* or *900*). 

Our model will be of the form:

$$ h(x) = \beta_0 + \sum_{j=1}^p x_j\beta_j $$

or if you just want to let x_0 to always be 1 then this can be simplified to:

$$ h(X) = \sum_{j=0}^p x_j\beta_j = \beta^Tx $$

Where $\beta$ and $x$ are both vectors and $\beta$ are the parameters **learned**.

### Cost Function

Now, given that we have a training set, we need to have our estimate $h(x)$, be as close as possible to the true values $y$. That is 
we want to minimize the error between the two for every observation. This leads us to our cost function:

$$ J(\beta) = \frac{1}{2} \sum^m_{i=1}(h_\beta(x^{(i)})-y^{(I)})^2 $$

