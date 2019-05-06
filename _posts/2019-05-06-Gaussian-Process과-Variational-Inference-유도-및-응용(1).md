---
title: Gaussian Process과 Variational Inference 유도 및 응용(1)
use_math: true
classes: single
layout: wide
---
*본 자료는 다음을 주로 참고하였습니다.*  
- PRML Chap 6. Kernel Methods
- Edwith 최성준님 강의, Bayesian Neural Network
- [Weight Uncertainty in Neural Networks, 2015](https://arxiv.org/abs/1505.05424)
- [VIME, 2016](https://arxiv.org/abs/1605.09674)
- [TD-VAE, 2019](https://arxiv.org/abs/1806.03107)  

**Introduction**  
Gaussian Process과 Variational Inference는 uncertainty in deep learning, variational auto-encoder, bayesian neural network 등의 이론적 기반이다.
머신 러닝의 확률론적인 적용에 관심이 있다면 반드시 탄탄하게 알아두어야 한다. 여기서는 Gaussian Process, Gaussian Process와 Variational Inference의 연관성, 
Bayesian neural network까지 이어서 설명하려 한다. 이 후 강화 학습 분야의 적용 예시을 알아보려 한다. 
Exploration, Curiosity 쪽 분야의 클래식한 논문인 VIME과 얼마 전 ICLR에 발표된 TD-VAE이다.  
  
**0. Gaussian Process과 Gaussian process regression**  

- Gaussian process의 정의 자체는 단순하다.
  - A gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution
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
  - $p(y)$$는 앞서 구했듯이 $$N(y \mid 0, K)$$이다. 
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
    
**1. Variational Inference**  
- 
