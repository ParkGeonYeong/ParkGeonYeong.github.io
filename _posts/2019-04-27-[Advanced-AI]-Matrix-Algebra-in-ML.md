---
title: [Advanced AI] Matrix Algebra in ML
use_math: true
classes: single
layout: wide
---
**Post for personnel summarizing**  
**All the contents in here is very important for various ML algorithm**  
  
- Data matrix
  - $$X = [x_1, ..., x_n] (pxn)$$
  - $$U'X = [U'x_1, ..., U'x_n]$$, each column vector of U is unit vector
    - If they are orthogonal, it can be regarded as projection of data X onto coordinate U  
    
- Range, Null-space
  - for A: mxn,
    - range of A: $$R(A) = Ax for some x in n-dimension space$$
      - Exactly same as column space
    - row space of A : column space or range of A transpose
    - null space of A : $$N(A) = x for all x in n-dimension space makes Ax=0$$
      - null space and row space are orthogonal complement
      - So that all linear combination of row vectors are orthogonal to any vector included in null space
    - rank of A : dimension of row or column space of A
      - It is the number of linearly independent rows/columns
      - $$dim(null(A))+rank(A)=n$$, because row and null space are orthogonal complement
- Positive definite  
  - Have many great properties for linear problem solving (Including machine learning)
  - For symmetric matrix A, it is positive definite if $$x'Ax>0$$ for all non-zero vector x
    - Positive semi-definite (PSD) : $$x'Ax{\geq}0$$
      - If A is positive definite, it is invertible because all eigen-values are positive
      - Because A is square, we can define trace of A
        - It is proportional to the circumference of the hyperillipsoid (More details later)
      - Because A is square, we can define determinant of A
        - Intuition of determinant : It can be thought of as the volume of the region by applying A to the unit cube
        - = the volume of the hyperilipsoid
        - Determinant of positive definite matrix : $$|A| = \prod_{i=1}^{p}{\lambda}_i$$, product of eigen-values
        - Because positive definite matrix A can be diagonalized as $$A=UDU'$$ which U is orthogonal-eigen matrix
          - $$|U|=|U'|=1$$
          - [More details](https://parkgeonyeong.github.io/Gaussian-Distribution-(1)/)
        - If there is all-zero rows or columns, or identical, or linear combination of each other, $$|A|=0$$
        - Matrix determinant lemma: $$|A+UV'|=|I+V'A^{-1}U||A|$$
          - [proof](https://en.wikipedia.org/wiki/Matrix_determinant_lemma)
- Covariance
  - Covariance of ith class: $$\sum_i =E(x-\mu_i)(x-\mu_i)'$$
  - Sample covariance of ith class: $$S^{i}_{sample}=\frac{1}{N}\sum^{N}{(x-\mu_i)(x-\mu_i)'}=XX'$$
  - But it's biased value (N should be N-1)
  - Covariance of y=CX: $$C{\sigma}C'$$ which $$\sigma$$ is covariance matrix(pxp) of X
    - Consider that $$\mu_y = C\mu_x$$
  - $$x'x$$ : Squared distance from the origin
  - $$x'{\sigma}x$$ : Squared statistical distance which is weighted by variance and covariance term
  - Similarly, $$x'Ax$$ is called as Quadratic form
    - It is so, so frequently expressed in terms of loss function of various linear ML methods
    - PCA, LDA, CCA, SVM, ...
  - More details later
- Eigen values and eigen vectors
  - Firstly we have to understands eigen values and eigen vectors
  - Eigen-value gives an understanding of a linear systems
  - If there exists an nx1 nonnull vector x s.t. $$Ax = {\lambda}x$$, x is called eigenvector
  - $$A-{\lambda}I$$ should not be invertible
    - For any particular eigenvector x, there exists only one eigenvalue $$\lambda=\frac{x'Ax}{x'x}$$
    - Eigenvector lies in null space of $$A-{\lambda}I$$
  - Eigen vector and similar matrix
    - Two nxn matrix X and Y are similar if for P, $$P^{-1}XP=Y$$
    - Because rank(XY)=rank(X), Y's rank is preserved
    - Similarly, determinant and trace is preserved
    - AND for eigenvector v of Y, 
      - $$Yv = {\lambda}v = P^{-1}XPv$$
      - $$X(Pv) = {\lambda}(Pv)$$
      - So X's eigen pair is gonna be $$(Pv, \lambda)$$
      - Eigen-value is preserved!
- SVD
  - If A is an mxn matrix, there exists the form of factorization $$A=UDV'$$
    - Which D is diagonal singular value matrix
    - U and V is orthogonal matrix, which U is eigenvector matrix of AA' and V is for A'A
    - Diagonalization is more strict than SVD
  - SO linear mapping of Ax can be decomposed as
    - 1. rotate by V'
    - 2. Diagonal scaled by D
    - 3. rotate back by U
  - Example: Shifting eigenvalues of symmetric matrix
    - $$A-{\lambda}I = UDU'-{\lambda}I = UDU'-{\lambda}UU' = U(D-{\lambda}I)U'$$
    - It scales down eigenvalues of A
    - Provides theoretical insight of Regularization [More details](Link)
  - Example: approximate matrix A by choosing partial-subset of whole eigenset
    - By choosing top-singular values, it can reduce error btw reconstructed and original matrices
    - Provides theoretical insight of PCA [More details](Link)
- Quadratic form
  - Very typical form in real-world problems and ML
  - 2-norm matrix, statistical distance, ...
  - IT HAS A UNIQUE MINIMUM
  - Usually represented as $$x'Ax$$
  - AND A CAN BE ASSUMED AS SYMMETRIC ONE W/O LOSS OF GENERALITY
    - Because $$x'Ax$$ is scalar, $$x'Ax=(x'Ax)'=x'A'x$$
    - $$x'Ax=\frac{1}{2}(x'Ax+x'A'x)=x'(\frac{A+A'}{2})x$$ which center term is exactly symmetric
  - Because A is symmetric, x' and x is orthogonal-eigen matrix
    - So after expansion, it exactly same as ellipsoid form
- Rayleigh quotient problem
  - Super important
  - Any kinda problem such as $$max_w{w'A'} s.t. ||w||=1$$
  - can be reformulated as rayleigh quotient $$max_w{\frac{w'Aw}{w'w}}$$
  - By Lagrangian Multiplier approach, it boils down to $$L(w,\lambda)=w'Aw-\lambda(w'w-1)$$
  - By partial derivation of w, it turns into $$Aw={\lambda}w$$ which is EIGENVALUE PROBLEM
  - More General form of rayleigh quotient: $$max_w{\frac{w'Aw}{w'Bw}}$$
  - One more time, it boils down into eigen value problem $$B^{-1}Aw={\lambda}w$$
  - But because there's no guarantee s.t. $$B^{-1}A$$ is symmetric, we have to make it symmetric
  - Because symmetric matrix have beautiful properties such as PSD and orthonormal eigenvalues
    - After some expansion, we can convert the original form into $$B^{-1/2}AB^{-1/2}v={\lambda}v$$