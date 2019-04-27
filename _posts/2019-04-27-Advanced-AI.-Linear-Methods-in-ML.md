---
title: [Advanced AI] Linear methods in ML
use_math: true
classes: single
layout: wide
use_math: true
---

**Post for personnel summarizing**  
**All the contents in here is very important for various ML algorithm**   
  
**0. Notation**   
- Ax=b
  - A describes our observation
    - (n,p) which n is number of observation, p is its dimension
  - x is a set of parameters (states of a system) to be estimated
  - b is a set of measurements (labels)
  
**1.Least Squares**  
- Goal : find x s.t. $$||Ax-b||^2=(Ax-b)^{T}(Ax-b)$$ is minimized
- Fundamental approaches of linear algebra
  - Let's note vector x from range R(A) as $$x_R$$, and from null space $$N(A^T)$$ as $$x_N$$
  - [For more details about range and null space](https://parkgeonyeong.github.io/2019-04-27-Advanced-AI-Matrix-Algebra-in-ML/)
  - Then, $$Ax=Ax_R+Ax_N$$, $$b=b_R+b_N$$
  - $$||Ax-b||^2 = ||Ax-b_R-b_N||^2 = ||Ax-b_R||^2+||b_N||^2$$ because $$Ax-b_R{\in}R(A)$$ is orthogonal to $$b_N{\in}N(A^T)$$
  - Because $$b_N$$ is regardless term with x, problems boils down into minimizing $$||Ax-b_R||^2$$
  - Suppose 'the answer' is $$x*$$ which minimizes given term so that $$Ax*=b_R$$. It's possible because both vector is in same linear space
  - Then, $$Ax*-b = Ax*-b_R-b_N=-b_N\inN(A^T)$$
  - Because now $$Ax*-b\inN(A^T)$$, $$A^T(Ax*-b)=0$$
  - So $$x*=(A^TA)^{-1}b$$, term inside paranthesis is known as pseudo-inverse of A
- Partial derivation (more easier)
  - Exactly same result can be driven from partial derivation of loss function w.r.t. x
- Intition 
