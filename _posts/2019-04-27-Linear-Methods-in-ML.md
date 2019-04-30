---
title: Linear methods in ML
classes: wide
layout: single
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
- $$||Ax-b||^2=(Ax-b)^{T}(Ax-b)$$
- Fundamental approaches of linear algebra
  - Let's note vector x from range R(A) as $$x_R$$, and from null space $$N(A^T)$$ as $$x_N$$
  - [For more details about range and null space](https://parkgeonyeong.github.io/2019-04-27-Advanced-AI-Matrix-Algebra-in-ML/)
  - Then, $$Ax=Ax_R+Ax_N$$, $$b=b_R+b_N$$
  - $$||Ax-b||^2 = ||Ax-b_R-b_N||^2 = ||Ax-b_R||^2+||b_N||^2$$ 
  - because $$Ax-b_R {\in} R(A)$$ is orthogonal to $$b_N {\in} N(A^T)$$
  - Because $$b_N$$ is regardless term with x, problems boils down into minimizing $$(Ax-b_R)^2$$
  - Suppose 'the answer' is $$x^*$$ which minimizes given term so that $$Ax^*=b_R$$. It's possible because both vector is in same linear space
  - Then, $$Ax^*-b = Ax^*-b_R-b_N=-b_N\inN(A^T)$$
  - Because now $$Ax^*-b\inN(A^T)$$, $$A^T(Ax^*-b)=0$$
  - So $$x^*=(A^TA)^{-1}b$$, term inside paranthesis is known as pseudo-inverse of A
- Partial derivation (more easier)
  - Exactly same result can be driven from partial derivation of loss function w.r.t. x
- Intition 
  - Ax means space exactly same as a span of column vectors A by each element x
  - Let A:(m,n) and n is linear independent basis of A. Then b(label) lies in m dimension where Ax lies in n sub-space
  - By reducing geometrical distance(euclidean-2-norm) btw two vectors, we can find projected value of b onto sub-space
  - It can be viewed as minimizing hyperplane-vector distance problem
  
**2. Derivative-based optimization**  
- What if we are not able to have nice linear-quadratic formed loss?
  - Or too complicated to solve?
  - For example, computes psuedo-inverse requires $$O(n^3)$$ complexity, and we have millions of data in real-world
- So we usually stick onto iterative algorithm to update parameters
  - MOST PRIMITIVE FORM OF LEARNING
- WE NEED GUARANTEE SUCH THAT UPDATE OF OUR MODEL ACTUALLY DECREASE LOSS TERM.
  - Let's show how gradient optimization can make it with taylor series expansion
- $$f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2}(x-a)^2+...$$
- Let f is our loss function, x is updated model parameter $$\theta+{\eta}d$$ which $$\eta$$ is learning rate, and a is $$\theta$$
- $$L(\theta+{\eta}d)=L(\theta)+{\eta}g^{T}d+...$$
  - Where g is derivation of L by $$\theta$$
  - **if we can set d such that $$g^{T}d$$ is negative, every update of $$\theta$$ by $${\eta}d$$ assures decrease of loss**
- What if we set $$d$$ as $$-Gg$$ such that G is positive definite matrix?
  - $$g^{T}d = -g^{T}Gg < 0$$ by definition of positive definite matrix
    - [Positive definite in previous post](https://parkgeonyeong.github.io/Matrix-Algebra-in-ML/)
- As a result, $$\theta_{t+1} = \theta_{t}-{\eta}Gg$$  
  
**3. Choosing G**  
- Steepest descent methods: $$G=I$$
- Newton's method: $$G=H^{-1}$$ which H is hessian matrix of L by $$\theta$$
  - Write above formula in slight different way, 
  - $$L(\theta_{t+1}) = L(\theta_t)+g^T(\theta_{t+1}-\theta_t)+\frac{1}{2}(\theta_{t+1}-\theta_{t})^{T}H(\theta_{t+1}-\theta_{t})+...$$
  - By derivate L with $$\theta_{t+1}$$, we can get $$g+H(\theta_{t+1}-\theta_{t})=0$$
  - $$\theta_{t+1} = \theta_{t}-H^{-1}g$$
- Levenberg-Marquardt modifications:
  - Put inverse of hessian is naive method not considering its invertibility
  - So by adding $${\lambda}I$$ to H, we can make it surely invertible
    - Because hessian matrix is symmetric, all we need to make it invertible is making its all eigenvalue positive
    - [Invertible symmetric matrix in previous post](https://parkgeonyeong.github.io/Matrix-Algebra-in-ML/)
- other methods: quasi-newton method, conjugate gradient methods, ...  
  
**4.Component analyses**  
4.1. Curse of dimensionality  
- Solving Ax=b in more smart way
  - Do not let dimension of x to be too complicated
  - In order to deal with curse of dimensionality, we can use linear component analyses as PCA, CCA, LDA
- Curse of dimensionality
  - High dimension, Low sample size problem (HDLSS) : d>>n
  - Problems of high dimension
    - data lie roughly on the surface of the high-dimensional sphere (Hard to see its true underlying pattern)
    - Distance among points get bigger, so that data is hardly clustered following its classes (See Euclidean distance equation)
    - Density of data goes lower, so that more data is needed to cover full range
      - Induce overfitting
4.2. PCA  
- Reduce data dimensionality, denoise data, get compact representation of data in low space
- **Idea : finds a subspace W in a way that maximizes the variance of the data projected onto W**
- Derivation
  - Centering data: $$X = (I-\frac{1}{n}11^{T})X$$
    - WHY CENTERING IS NEEDED IN PCA?
      - data should be passed by origin of coordinates so that W can be some subspace of original coordinates
      - Covariance of projected data can be easily described (in next step)
  - Project onto w: $$X_{new} = w'X$$
  - Conditioning w so that covariance of projected data can be keeped as much as possible
    - covariance of new data: $$X_{new}X_{new}^{T}=w'XX'w=w'Cw$$ which C is original covariance
    - ||w||^{2}=1
      - w/o loss of generality
  - Solve rayleigh quotient problem
    - $$max_w \frac{w^{T}Cw}{w^{T}w}$$
    - **It becomes eigenvector problem!**
4.3. CCA  
- Useful to evaluate linear dependency of two high-dimensional variables in lower-denoised dimension
- Find a low-dimensional project of X and Y such that they are maximally correlated
- **Idea : Maximize correlation of projected data $$corr(X_a, X_b)$$ by each projection vector $$w_a,w_b$$**
- $$max_{w_a,w_b}\rho = max \frac{w_a^{T}C_{ab}w_b}{\sqrt{w_a^{T}C_{aa}w_a}\sqrt{w_b^{T}C_{bb}w_b}}$$
  - which $$C_ab$$ is covariance btw original X, Y and $$C_aa, C_bb$$ is autocovariance of X, Y
  - It boils down to largrange multiplier technique, with constraint $$w_a^{T}C_{aa}w_a=1$$ and $$w_b^{T}C_{bb}w_b=1$$
  - **So it is generalized eigenvalue problem**  
  
4.4 LDA  
- Discrimination algorithm
  - Classification : given training dataset and its class, classify next incoming data into specific class
  - Discrimination : given distinct set of dataset, separate it and assign next incoming data into specific population
- **Idea : project the data so that the projected data among classes are maximaly separated**
  - Maximizing separability of distinct data in the projected space
- $$argmax_{w}J(w)=argmax_{w}\frac{S_{w}^{B}}{S_{w}^{W}}$$
  - B: scatterness Between-class, W: scatterness within-class
  - Substitute original data as w-projected one, it becomes as $$argmax_{w}\frac{w^{T}S^{B}w}{w^{T}S^{W}w}$$
  - **It is perfectly generalized as rayleigh quotient problem**
    - By lagrange multiplier technique, it becomes as $$S^{(W)-1}S^{B}w={\lambda}w$$
    - What if $$S^{W}$$ is not invertible?
      - Use regularized LDA
4.5 Summary  
- Sometimes high-dimensional data should be reduced to lower-one
- Several different component analysis can be exploited by data property
  - And they all can be driven by eigen-value problems
  - However, the role of each eigen-vector is different
  - For example, eigenvector in PCA is informative for whole dataset characteristic, while the one for LDA is more about information useful for discriminating different dataset maximally.
