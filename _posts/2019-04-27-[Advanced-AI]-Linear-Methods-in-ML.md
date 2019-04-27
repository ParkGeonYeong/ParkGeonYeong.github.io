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
  
**1.Various derivation of Least Squares**  
- Goal : find x s.t. $$||Ax-b||^2=(Ax-b)^{T}(Ax-b)$$ is minimized
- Fundamental approaches of linear algebra
  - Let's note vector x from range R(A) as $$x_R$$, and from null space $$N(A^T)$$ as $$x_N$$
  - [For more details about range and null space]( 
