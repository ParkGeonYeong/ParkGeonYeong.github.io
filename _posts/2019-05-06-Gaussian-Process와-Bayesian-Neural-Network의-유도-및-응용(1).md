---
title: Gaussian Process와 Bayesian Neural Network의 유도 및 응용(1)
use_math: true
classes: wide
layout: single
---
*본 자료는 다음을 주로 참고하였습니다.*  
- PRML Chap 6. Kernel Methods
- Edwith 최성준님 강의, Bayesian Neural Network
- [Weight Uncertainty in Neural Networks, 2015](https://arxiv.org/abs/1505.05424)
- [VIME, 2016](https://arxiv.org/abs/1605.09674)
- [TD-VAE, 2019](https://arxiv.org/abs/1806.03107)  

**Introduction**  
Gaussian Process과 Variational Inference는 uncertainty in deep learning, variational auto-encoder, bayesian neural network 등의 이론적 기반이다.
머신 러닝의 확률론적인 적용에 관심이 있다면 반드시 탄탄하게 알아두어야 한다. 여기서는 Gaussian Process, Variational Inference을 먼저 짚고 
Bayesian neural network까지 이어서 설명하려 한다. 특히 variational inference, bayesian neural network는 최근 강화 학습 분야에도 활발히 적용되고 있기 때문에 그 예시를 알아보려 한다. 
Exploration, Curiosity 쪽 분야의 클래식한 논문인 VIME과 얼마 전 ICLR에 발표된 TD-VAE가 그것이다.  
  
**0. Gaussian Process과 Gaussian process regression**  
- Gaussian process의 정의 자체는 단순하다.
  - **A gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution**
  - 그러나 그 개념을 regression에 적용하면 더 유용하다. 
  - 가령 $$y=w^{T}\phi(x)$$의 regression 식을 생각하자. 이때 $$p(w)=N(w \mid 0, \sum_p)$$의 prior를 도입하면 $$y$$는 linear combination of
  gaussian distributed variables이 되어, 그 자신도 gaussian이 된다.
  - 즉 학습 데이터를 활용해 추론한 regression의 output도 gaussian process을 이룬다는 것이다.
  - Gaussian y에 대해서, $$E(y)=0$$, $$Cov(y) = E(yy^T) = \phi(x)^{T}E(ww^T)\phi(x') = \phi(x)^{T}\sum_p\phi(x')=k(x,x')$$이다.
    - $$k$$는 kernel로써, high dimensional space에서 두 벡터의 similarity(Gram matrix)를 의미한다. 
    - Regression에서 Training data 간의 linear correlation이 높으면, training output 역시 높은 correlation을 갖는다. 
    Kernel을 통해 이를 고려한 gaussian process를 만들 수 있다. Kernel의 종류에 따라 gaussian process의 smoothing 정도 역시 달라진다. [Kernel post 준비 중]()
    - ![image](https://user-images.githubusercontent.com/46081019/57219947-a035c700-7034-11e9-9ed2-de0a1be874cf.png)  
    - Training data을 이용해 Gaussian process을 구축하고 샘플링을 거쳤다. 이때 gaussian kernel(왼)과 exponential kernel(오)를 활용한 결과가 다른 점을 확인할 수 있다.
  - 지금까지 주어진 데이터 x와 커널, prior weight w를 이용하여 y를 gaussian으로 표현했다.
    - 이를 regression function f에 대해 수식으로 쓰면 다음과 같다.
    - $$y=f(x)~GP(m(x), k(x,x'))$$
    - 이때 m(x), k(x,x')은 각각 gaussian process의 mean과 covariance이다.
  - 만약 여기에 새로운 데이터 $$x^*$$을 추가한다면, $$x^*$$에 대한 output "distribution" $$f(x^*)$$까지도 얻을 수 있을 것이다. 이를 gaussian process regression이라 한다.
  - 즉 임의의 추가 데이터에 대해서, 기존 데이터의 gaussian process를 활용해 predictive distribution of output을 얻을 수 있다.
  - 수식을 살펴보기 위해 다시 regression으로 돌아가자. 이번에는 (input=y, output=t) 데이터에 gaussian noise $$\epsilon$$를 부여한다. 
  - $$t_n=y_n+\epsilon_n$$
  - 각 data point의 noise는 independent하므로 covariance는 diagonal하다. 따라서 $$p(t \mid y)=N(t \mid y, \beta^{-1}I_N)$$이라 할 수 있다.
  - $$p(y)$$는 앞서 구했듯이 $$N(y \mid 0, K)$$이다. 
  - 따라서 bayes rule에 의해 $$p(t)=\int{p(t \mid y)p(y)dy} = N(t \mid 0, C)$$이다.
    - $$C(x_n, x_m) = k(x_n, x_m)+\beta^{-1}\delta_{mn}$$
    - 이때 $$p(t \mid y), p(y)$$의 randomness의 근원인 noise와 k는 서로 관련 없기 때문에 covariance 역시 단순히 summation으로 나타낼 수 있다.
  - 이제 output을 input variables의 joint distribution model로 표현했다. 
    - 즉 데이터를 기반으로 non-parametric하게 regression function f를 근사한 것이다.
  - 지금까지의 output을 모은 vector를 $$t$$이라 하자.
  - 새로운 $$n+1$$번째 데이터가 들어왔을 때, 우리의 regression을 이용해 새로운 joint distribution $$p(t_{n+1})= N(t_{n+1} \mid 0, C_{n+1})$$을 구해야 한다.
  - **이를 conditional하게 표현하면 $$p(t_{n+1} \mid t)$$를 구하는 것과 같다.**
    - 이때 $$t$$는 이미 gaussian이므로, $$p(t_{n+1} \mid t)$$을 conditional gaussian distribution으로 표현할 수 있다.
    - Conditional gaussian distribution은 PRML의 2.3.1절에 나와있다.
    - 조건부 gaussian의 mean, covariance를 이용해 새로운 gaussian의 mean, covariance를 다음과 같이 유도 가능하다.
    - ![image](https://user-images.githubusercontent.com/46081019/57221399-11c44400-703a-11e9-8545-bdb1fd32904b.png)  
    - $$f_*,x_*,X,f$$는 각각 새로운 output, input, 기존의 Input, Output을 의미한다.
    - 증명은 포스트 가장 밑의 appendix 참조
    - 참고로 kernel matrix에 data noise term이 더해져, inverse matrix가 반드시 존재하도록 되어 있다.
    - 이는 symmetric한 gram matrix를 확실히 positive definite matrix로 만들었기 때문이다.
    - 따라서 closed-form answer가 존재한다.
  - 이로써 우리는 주어진 데이터를 기반으로 gaussian process를 만들었고, 
  아직 보지 못한 새로운 데이터에 대해 predictive distribution $$f_* \mid x_*,X,f$$을 얻었다. 
  - 이때 식을 유심히 보면 우리의 가장 큰 관심사인 $$mean(f_*)$$가 $$x_*$$와 기존 데이터 $$X$$의 kernel 값에 대한 linear combination으로 이루어져 있는 것을 알 수 있다.
    - 따라서 predictivemean of a gaussian process는 $$f_* = \sum_{i=1}^{n}{a_ik(x_i,x_*)}$$으로 쓸 수 있다.
    - 학습 데이터가 많으면 많을 수록 새로운 데이터에 대해 정확한 predictive mean을 추론할 수 있다.
  - **정리하면, gaussian process regression은 우리의 관심사인 regression function f를 무한한 공간에서 정의하는 non-parametric bayesian regression method이다. Regression 차원에 대한 어떠한 prior determination 필요 없이, finite training set을 기반으로 test set에서 새로운 inference를 얻을 수 있다.**
    - 다만 몇 백만개의 데이터로 이루어진 gram matrix의 inverse를 계산하는 복잡도가 굉장히 높기 때문에, 최근 데이터셋에 바로 적용하기에는 무리가 있다.
    - 이를 해결하기 위해 bayesian neural network를 사용할 수 있는데, 뒤에 이어서 설명하고자 한다.
  - 또한 regression function space로 gaussian process regression을 설명하지 않고, regression의 weight space에서부터 설명하는 방식이 있다.
    - 그러나 복잡한 유도 과정 끝에 얻는 결과는 동일하다.  
    
**0.1 Gaussian Process와 Bayesian Neural Network의 관계**    
- 결론적으로 Bayesian Neural Network는 weight을 prior distribution으로 나타낸 neural network이다.
  - 이를 통해 single point prediction이 아닌, predictive distribution을 얻는다.
- 왜냐하면 weight를 distribution으로 표현하면 이론적으로 network는 gaussian process에 수렴하기 때문이다.
  - 이와 관련된 유명한 이론으로 [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)이 있는데, two hidden layer로 어떤 함수도 근사가 가능하다는 이론이다.
  - Universal approximation theorem이 gaussian process regression에도 적용되는 것이다.
- $$f(x)=b+\sum_{j=1}^{N_H}v_jh(x;u_j)$$으로 two-layer neural network를 나타내자. **이때 $$f(x)$$는 independently weighted random variable의 합이므로 central limit theorem에 따라 gaussian process**로 근사된다.

**1. Variational Inference**  
- Gaussian process를 통해 infinite dimensional function space에서 특정 f를 얻을 수 있다.
- 이 함수 f로 우리가 원하는 어떤 posterior distribution $$p(w \mid D)$$를 근사해야 한다고 해보자.
  - 어떤 함수를 바로 구하고 싶다면 numerical하게 계산하는 게 가장 이상적인 방법일 것이다.
  - 그러나 latent variable이 너무 많다던지, dimensionality가 너무 높다던지, 다양한 이유로 이 posterior distribution을 구하기 어려울 수 있다.
  - 이런 경우 우리는 어쩔 수 없이 posterior distribution을 analytical하게 근사하게 된다.
  - 이때 사용할 수 있는 테크닉이 variational Inference라고 할 수 있다. 
- Intractable한 posterior distribution의 '값을 구하는' 또 다른 방법으로는 stochastic sampling(MCMC) 등이 있다.
  - [Previous MCMC post](https://parkgeonyeong.github.io/Markov-Chain-Monte-Carlo%EC%99%80-Posterior-Sampling/)
  - 그러나 computationally demanding하고, required distribution을 직접적으로 알아낼 순 없다.
- Variational inference를 위해서는 우선 varitional distribution $$q$$를 잡아야 한다.
- $$q$$의 범위는 constrained되어야 하는데, 가령 quadratic form이나 linear combination of fixed basis functions 등이 있겠다.
- 우리가 관측하지 못하는 latent variable을 $$Z$$, 관측한 observation을 $$X$$이라 하자.
- 우리의 목표는 model evidence $$p(X)$$를 최대한 높이는 것이다. Log-likelihood를 수식으로 전개하면 다음과 같다.
- $$\begin{align} lnP(X) &= \int{q(w)lnP(X)dw} \\
&=\int{q(w)ln\frac{P(w \mid X)P(X)}{P(w \mid X)}dw} \\&= \int{q(w)ln\frac{q(w)P(w, X)}{q(w)P(w \mid X)}dw} \\&=\int{q(w)ln\frac{P(X \mid w)p(w)}{q(w)}dw}-\int{q(w)ln\frac{P(w \mid X)}{q(w)}dw} \\&= 
\int{q(w)ln\frac{P(X \mid w)p(w)}{q(w)}dw}+KL(q \mid\mid p(w \mid X))\end{align}$$
- 이때 KL-divergence term은 positive하므로 남은 항은 자연스럽게 model evidence의 lower bound가 된다. 즉 이 lower bound term를 maximize하는게 전체 inference 과정을 보장한다. 
  - 이를 variational free energy, 혹은 ELBO라고 부르며 variational inference의 핵심 loss라고 할 수 있다.
  - 보통은 -를 붙여서 minimize free energy라고 한다.
  - 물론 lower bound를 가장 tight하게 하는 방법은 KL divergence term을 0으로 만드는 것이지만, true posterior distribution을 모르기 때문에 불가능하다.
  - lower bound를 maximize하는 과정에서도 KL divergence term은 등장하며, 그 크기가 줄어들어야 하는 regularizer 역할을 한다.  
  
**1.1 Factorized distributions**  
- 이후에 설명하겠지만 Bayesian neural network에서 Variational distribution을 정의하는 방식으로는 diagonal gaussian distribution을 많이 사용한다.
- 여기에서는 가장 기초적으로 많이 사용되는 방식인 factorized distribution을 예시로 들겠다.
  - $$q(z)=\prod_{i=1}^{M}q_i(Z_i)$$
- Individual latent variable $$Z_i$$에는 가능한 분포의 제약이 딱히 없다. Elements of Z를 disjoint group으로 표현한 것이다.
- 이때 앞서 위에서 구한 ELBO term을 위의 variational distribution으로 적으면 다음과 같다.
- ![image](https://user-images.githubusercontent.com/46081019/57233231-34b22080-7059-11e9-9763-dc64ecd0308c.png)  
- 식이 임의의 j-th distribution $$q_j$$에 대해 정리되었으며, 나머지는 상수항 처리가 되었다.
- 첫 번째 항에서 $$lnp(X, Z_j)$$은 jth term을 제외한 나머지에 대해 구한 model evidence의 expectation이다.
  - 직관적으로 $$lnq_j$$이 이 term과 가까워질수록 $$q_j$$가 jth latent variable에 대해 잘 근사했다고 할 수 있겠다.
  - 수식적으로는 두 항이 $$KL(q_j \mid\mid p(X, Z_j))$$ 관계이기 때문에 그렇다.
- 즉 j에 대한 optimal solution $$q_j^*$$은 다음과 같다.
  - ![image](https://user-images.githubusercontent.com/46081019/57233627-f5d09a80-7059-11e9-9c86-10f7f320c090.png)  
  - 이때 이 optimal solution은 다른 latent variables과 entangled된 상태에서 구했기 때문에 explicit solution이라고 볼 수 없다.
  - 따라서 이를 피하기 위해 iterative한 solution이 필요한데, 가령 모든 i에 대해 $$q_i(Z)$$를 initialize하고, 각 latent factor를 순환하면서 
  계속해서 revised estimate으로 업데이트한다. 이를 수렴할 때까지 반복한다.
- Variational Inference에는 True-posterior distribution이 multi-modal일 때 발생하는 단점이 존재한다. 이를 Mode collapse라고 한다.
- ![image](https://user-images.githubusercontent.com/46081019/57234254-3381f300-705b-11e9-9b74-7cc470f1410d.png)  
- 가령 blue multi-modal gaussian function P가 있고, 이를 두 latent variable $$z_1, z_2$$를 이용한 gaussian distribution의 곱으로 표현하고자 한다.
  - 이때 $$KL(q \mid\mid P)$$를 최소화하는 과정에서, q는 P가 낮은 곳에 value를 assign하지 않는다.
  - 따라서 두 mode 중 하나에 빠져, KL divergence의 local minimum을 취하게 된다.
