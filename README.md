# CNFs for amortized variational inference (+ other stuff)

This repo is an exploration of some ideas surrounding continuous normalizing flows (CNFs) and their application to variational inference in probabilistic programs. 

## Variational inference (VI)

Let's refer to our latent space as $Z$ and our observed space as $X$.

Suppose we have a forward model $p(x|z)$. Our aim is to learn $p(z|x)$, for which purpose we minimize either $KL(q(\cdot | x)|p(\cdot | x))$ (standard VI) or $KL(p(\cdot | x)|q(\cdot | x))$ (inference compilation). Here $q$ is a model of our choosing parametrized by some $\theta$.

Inference compilation can be achieved with steps of the following expected gradient:

$$
- E_{p(x,z)}\llbracket \nabla_\phi \log q_\phi(z|x) \rrbracket
$$

(Easy to derive, see section 3.1 of [this paper](https://arxiv.org/pdf/1610.09900.pdf))



## What are normalizing flows

[Normalizing flows](https://deepgenerativemodels.github.io/notes/flow/) are a technique where you push forward a simple distribution through a differentiable bijective map (the normalizing flow) to obtain a complex distribution whose PDF can be obtained via through the determinant of the Jacobian of the map. Computing this determinant tends to be a computational bottleneck.

For example, given a flow $f : X -> Z$, and $p(x)$, we can compute the probability density of some $z \in Z$ by multiplying the Jacobian determinant by $p(f^{-1}(z))$.

Conversely, we can sample from $Z$ by:

```js
x <- px
return f(x)
```

## Neural ODEs

Given an ODE $\frac{dy}{dt} = f(y,t; \theta)$ there is a unique "flow" that evolves $y$ from any initial point $y(0)$ to $y(t)$, and which can be approximately obtained by a numerical solver. 

With respect to deep learning models, the idea is that we learn $f$. The [original neural ODE paper](https://arxiv.org/pdf/1806.07366.pdf) shows how to do this without differentiating through the solver. 

## Continuous normalizing flows

Since the time-evolving map implied by $f$ is a differentiable bijective map (*whether or not $f$ is bijective*), we can use it as a normalizing flow.

An advantage is that we don't need to repeatedly compute the determinant of the Jacobian, since thanks to Jacobi's formula, we have:

$$
\frac{\partial log(p(y(t)))}{\partial t} = -tr\llbracket \frac{df}{dy(t)}\rrbracket
$$

So we can compute the determinant once, obtain the initial log probability, and then evolve it forward via a numerical solution to this ODE, computing the cheaper trace of a Jacobian.

## (C)NFs for VI

In the context of VI, we can define $q_\theta$ via a normalizing flow and thus obtain a much more expressive family of distributions than the more common choice of setting $\theta$ as the parameters of some well known distribution (e.g. the mean and variance of a normal).

For CNFs, we define $f$ parametrized by $phi$ and $x$:

$$ 
\frac{dz}{dt} = f_\phi(z,t; x)
$$


and then push some $q_0$ through the flow obtained from this ODE. We can calculate an expectation over $\log q(z|x)$ by Monte Carlo.

## Project idea

Can we generalize CNF-style variational inference to arbitrary probabilistic programs? This would be achieved by fusing the very general approach of [Inference Compilation and Universal Probabilistic Programming](https://arxiv.org/pdf/1610.09900.pdf) with CNF-VI, as in [Structured Conditional Continuous Normalizing Flows for Efficient Amortized Inference in Graphical Models](http://proceedings.mlr.press/v108/weilbach20a/weilbach20a.pdf)

### Challenges

- Discrete variables

- Arbitrary stochastic control flow + recursion, i.e. how to extend to universal PPLs
