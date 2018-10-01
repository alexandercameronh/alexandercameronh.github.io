---
title:  "Linear Regression From Scratch"
---

The objective of this post is simply just to implement linear regression from scratch using Python. I won't dive into the math behind it all and so I'll save that for another post.

Since it is a regression problem, $y$ is a real-valued output (e.g. *5.23* or *900*). 

Our model will be of the form:

$$ h(x) = \theta_0 + \sum_{j=1}^p x_j\theta_j $$

or if you just want to let $x_0 = 1$ then this can be simplified to:

$$ h(x) = \sum_{j=0}^p x_j\theta_j = \beta^Tx $$

Where $\theta$ and $x$ are both vectors and $\theta$ are the parameters **learned**.

#### Cost Function

Now, given that we have a training set, we need to have our estimate $h(x)$ be as close as possible to the true values $y$. That is 
we want to minimize the error between the two for every observation. This leads us to our cost function:

$$ J(\theta) = \frac{1}{2} \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(I)})^2 $$

#### Least Mean Squares

We want to chose $\theta$ such that our cost function is **minimized**. This is done by using an algorithm that initially sets $\theta$ to some value and then repeatedly makes changes to $\theta$ and hopefully converges to a value that minimizes $J(\theta)$.
This is done with **gradient descent**.

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_j)$$

":=" is an assignment operator, where the \[updated\] $\theta_j$ is assigned to everything on the RHS.

$\alpha$ is the learning rate.


