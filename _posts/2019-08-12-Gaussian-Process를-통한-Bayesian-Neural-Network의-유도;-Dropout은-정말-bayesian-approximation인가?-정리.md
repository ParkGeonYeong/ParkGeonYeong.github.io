---
title: Gaussian Process를 통한 Bayesian Neural Network의 유도; Dropout은 정말 bayesian approximation인가?
use_math: true
classes: wide
layout: single
---

본 자료는 다음을 주로 참고했습니다.   
- [Weight Uncertainty in Neural Networks](https://openreview.net/pdf?id=Sy2fzU9gl)
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
- [Reddit Discussion](https://www.reddit.com/r/MachineLearning/comments/7bm4b2/d_what_is_the_current_state_of_dropout_as/)
- [Ian Osband's Note on risk and uncertainty](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf)  
  
  
이전 포스트와 이어집니다. 
- [Gaussian Process와 Variational Inference](https://parkgeonyeong.github.io/Gaussian-Process%EC%99%80-Variational-Inference/)  
  
  

Bayesian Neural Network은 weight를 probabilistic distribution으로 표현함으로써 uncertainty를 estimate할 수 있는 모델로써, 
흔히들 "모르는 것을 모른다고 말하는" 모델이라 부른다. 활용 방안과 가치, 개략적인 소개 등은 다양한 자료가 존재하므로 대체한다.  
- [NIPS 2016 BNN 워크샵](https://www.youtube.com/channel/UC_LBLWLfKk5rMKDOHoO7vPQ)
- [Taeoh Kim님의 블로그 포스팅](https://taeoh-kim.github.io/blog/bayesian-deep-learning-introduction/)    

여기서는 bayesian neural network의 중요한 선구자적 기반이 되는 두 논문 Blundell et al., "Weight Uncertainty in Neural Networks"과 
Gal et al., "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"을 수식 중심으로 구체적으로 
살펴보고, 구현체도 참고한다. [Bayes by Backprop 구현체](http://krasserm.github.io/2019/03/14/bayesian-neural-networks/)
  
이후 소개할 내용은 "과연 Dropout이 정말 true posterior distribution을 잘 근사하는가?"에 대한 discussion과 관련 paper이다. 
Dropout 논문 발표 이후, Deep exploration via bootstrap이나 distributional RL 시리즈 등으로 유명한 스탠포드의 Ian Osband을 필두로 
dropout과 bayesian approximation에 대한 논의가 이뤄진 것으로 보인다. 이를 살펴봄으로써 Bayesian Neural Network의 본질적 고민과 
특성에 대해 보다 깊게 이해할 수 있을 것이다.   
  
  
**1. Weight Uncertainty in Neural Networks**  
- Bayes by Backprop 혹은 BBB라 불린다고도 한다. 
  - 제목 그대로 weight를 parameterize하여 uncertainty를 부과한다.
  - 또한 Kingma의 reparameterization trick을 활용해 backprop으로 학습한다.
- 기존의 많은 NN은 대부분 deterministic layer이자 weight의 posterior point estimates으로 구성되어 있다.
  - 반면 weight를 "분포"화하면, weight의 sampling에 따라 ensemble of networks를 학습시킬 수 있다.
  - ![image](https://user-images.githubusercontent.com/46081019/63660859-44fe2f00-c7f3-11e9-9e01-623e7c029d4c.png)  
    - 여기까지 보면 하고자 하는 목표가 dropout과 비슷하다.
    - 실제로 그 다음 논문은 weight의 posterior 분포를 dropout으로 근사한다.
  - 이를 통해 data sample에 대한 uncertainty을 quantify할 수 있고, overfitting을 피할 수 있다. 
  - 여기서는 NN의 앙상블을 학습시키면서도 weight를 mean과 diagonal covariance $$\mu, \rho$$으로 근사하기 때문에, 파라미터가 크게 늘어나지 않는다.
    - 그 다음 논문에서는 이것도 많다며 아예 추가 파라미터를 도입하지 않는다.
- **이렇게 weight의 분포를 잡으면 문제는 MLE에서 MAP으로 바뀐다.**
  - $$W^{MLE} = argmax_{w} logP(D \mid w)$$
  - $$W^{MAP} = argmax_{w} logP(w \mid D) = argmax_{w} logP(D \mid w) + logP(w)$$
- 이 posterior weight distribution은 당연히 intractable하다.
  - $$P(w \mid y, x) = \int P(y \mid x, w)P(w)dw$$
  - 여기서 우측 적분식의 likelihood 항은 N개의 데이터, J의 weight dimension에 대해 표현되는 product distribution 식이다. 
  - 이런 고차원의 distribution에 대해 integration하는 것은 현실적으로 불가능하다.
- **따라서 여기서는 variational approximation을 통해 true posterior distribution을 근사한다.**
  - 앞서 언급한 parameter of weights $$\mu, \rho$$를 벡터 $$\theta$$로 생각하자.
  - 이때 우리의 variational approxiation $$q$$와 true distribution $$p$$ 사이의 $$KL-div$$를 최소화해야한다.
  - $$\begin{equation} \begin{split}
  \theta^* &= argmin_{\theta}KL[q(w \mid \theta) \parallel p(w \mid D)] \\
  &= argmin_{\theta} \int q(w \mid \theta) log \frac{q(w \mid \theta)}{p(w)p(D \mid w)}dw \\
  &= argmin_{\theta}KL[q(w \mid \theta) \parallel p(w)] -E_{q(w \mid \theta)}[logp(D\mid w)] 
  \end{split}
  \end{equation}$$
- weight으로 표현되어 있으나 꼴은 VAE와 비슷하다.
  - 아직 고차원의 weight에 대해 평균과 KL-div가 포함되어 있는, 계산이 쉽지 않은 꼴이다.
  - 논문에서는 이를 reparameterization trick으로 풀어낸다.
- ![image](https://user-images.githubusercontent.com/46081019/63661515-dd95ae80-c7f5-11e9-8a6f-1f8067de5fa8.png)  
- 따라서 최종적으로 얻는 cost는 복잡한 integration, KL, expectation 없이 weight의 Monte Carlo sampling만으로 표현된다.
  - $$F(D, \theta) = \sum_{i} logq(w_i \mid \theta) - logp(w) -logp(D\mid w_i)$$
  - VAE를 hidden node가 아니라 훨씬 고차원인 weight space에 대해 정의했다고 생각할 수 있다. 
  - 이렇게 posterior에서 뽑힌 특정 weight sample에 의해 loss를 표현할 수 있기 때문에 학습이 가능해졌다.
- 각 weight의 sample을 다음과 같이 표현할 수 있다.
  - $$w = t(\theta, \epsilon) = \mu + log(1 + exp(\rho)) \circ \epsilon$$ 
  - 이때 $$\epsilon$$은 reparameterization trick의 noise이다.
- 이제 위의 Thm과 $$F(D, \theta), w$$를 이용해 학습하면 다음과 같다.
  - ![image](https://user-images.githubusercontent.com/46081019/63662022-e38c8f00-c7f7-11e9-96b6-db2b7fd2a7b8.png)  
- 논문에서는 학습 과정에서 두가지 디테일한 트릭을 사용했다.
  - Mixture Prior: Prior를 단순히 $$N(0, I)$$을 쓰지 않고 평균이 0인 Gaussian Mixture를 사용했다.
    - $$P(w) = \prod_j \pi N(w_j \mid 0, \sigma_1^2) + (1 - \pi) N(w_j \mid 0, \sigma_2^2)$$
    - 이때 $$\sigma_1$$이 다른 $$\sigma_2$$보다 크고, $$sigma_2$$는 0에 가깝도록 설정한다. 이를 통해 하나의 단순한 prior stddev가 weight에 공유되는 것을 피한다. 또한 특정 weight는 sparse하게 코딩되는 것을 강제할 수 있다. 
  - KL re-weighting: Prior와의 KL divergence은 variational posterior distribution의 성능을 과하게 억제할 수 있다. 따라서 학습 초반에는 Prior loss term의 weight를 많이 낮췄다가 점차 증진해간다.
  - 다른 논문에서는 비슷한 아이디어를 KL annealing이라는 표현으로 사용한다. VAE의 posterior collapse에 대항함.
  - [Generating Sentences from a Continuous Space](https://arxiv.org/pdf/1511.06349.pdf)
  
- ![image](https://user-images.githubusercontent.com/46081019/63664561-f73cf300-c801-11e9-95cf-021fcfe2c322.png)  
  - Noisy data에 regression fitting시킨 모습. 왼쪽의 BBB을 활용하면 Uncertain한 region에 대해서 불확실한 값을 리턴한다.

**2. Dropout as Bayesian Approximation**  
- 논문을 한 줄로 요약한다면 **dropout의 적용만으로 이론적으로 gaussian process의 bayesian inference을 근사할 수 있다**는 것이다.
  - **디테일하게는 Bernoulli dropout만으로도 Gaussian Process의 integration over weight space을 근사할 수 있다고 주장한다.** 
  - 여기서는 NN의 W1, W2, b를 이용해 GP의 PSD kernel을 정의한다. 
  - Kernel을 도입함으로써 NN의 $$p(y \mid x, w)$$을 GP로 만들 수 있음을 보였다.
  - **이후 Intractable True Weight Posterior를 ELBO로 학습한다**.
  - 이 과정에서 dropout과 reparameterization trick(Kingma et al.,)을 통해 ELBO의 계산량을 확 줄인다.
  - 정확히는 Monte-Carlo dropout만으로도 ELBO를 얻을 수 있게 함!
  - 결국은 reparameterization trick을 두 번 쓴 셈이다.
  - 이를 통해 RL 실험에서는 posterior sampling을 근사하기도 함
  - 앞선 논문과의 차이점은, weight uncertainty를 mu와 stddev으로 parameterize하느냐, monte carlo sampling으로 parameterize하느냐의 차이이다.
    - Dropout을 도입하여 Gaussian Process의 weight space of view을 Neural Network 차원에서 해석한다.
- 논문의 흐름을 다시 요약하면 Weight을 Variational Parameter로 도입함으로써 Neural Network가 Gaussian Process으로 근사될 수 있음을 보이고, 그 다음 이 true weight parameter를 variational inference하는 과정이다. 
  - 우선 Gaussian Process을 정의하기 위해서는 Positive Semi-Definite한 kernel, 혹은 데이터 간 covariance의 정의가 필요하다. 
  - 여기서는 Neural Network을 사용하기 때문에 이 kernel을 $$K(x, y) = \int p(w)p(b)\sigma(w^{T}x+b)\sigma(w^{T}y+b)dwdb$$으로 표현
    - 이때 w, b, $$\sigma$$는 결국 spectral decomposition이 되어 나중에 NN의 weight, bias, non-linear activation이 된다.
- **만약 Gaussian process로 현재까지의 dataset (X, Y)와 앞으로의 data point (x, y)가 표현된다면 gaussian process의 weight-space view에 따라 다음 식이 성립해야 한다.** (Rasmussen, Williams 책의 Chap 2. 참고)
  - $$p(y \mid x, X, Y) = \int p(y \mid x, w)p(w \mid X, Y)dw$$
  - $$p(y \mid x, w) = N(y; \hat{y}, \tau^{-1}I_{D])$$
- 이때 이상적으로 $$\hat{y}$$는 우리 NN의 output이 되어야 함을 명심하며, 앞서 정의한 kernel이 어떻게 이 결론을 이끌어내는지 보자.
  - $$K(x, y) = \int p(w)p(b)\sigma(w^{T}x+b)\sigma(w^{T}y+b)dwdb$$
  - 이 커널은 Monte-Carlo integration estimation을 통해 w, b의 sampling으로 계산할 수 있다고 치자 ($$\hat{K}$$).
  - zero-mean Weight-space view의 Gaussian Process 정의에 따라 데이터 Y의 확률을 다음 GP로 근사한다.
    - $$F \mid X, W_1, b \sim N(0, \hat{K}(X, X))$$
    - $$Y \mid F \sim N(F, \tau^{-1}I_N)$$
      - 이때 $$\tau$$는 데이터의 noise 혹은 precision level이다.
    - GP의 kernel space에 projection된 vector $$\phi$$을 $$\phi(x, W_1, b) = \sqrt{\frac{1}{K}}\sigma(W_1^{T}x+b)$$이라 하자.
    - **그러면 kernel 혹은 covariance 정의에 의해 원래대로라면 우리의 GP prior $$p(Y \mid X) = \int N(Y; 0, \phi\phi^T + \tau^{-1}I)p(W_1)p(b)dW_{1}db$$가 되어야 한다.**
      - 여기에는 NxN짜리 covariance가 붙어있기 때문에 활용하기 좋지 않다. (N; 데이터 개수, D; 데이터 차원)
    - **따라서 이 NxD 짜리 GP 식을 다시 column-wise으로 해석해서 우리가 익히 알고 있는 형태로 바꿔보자.**
      - **원래는 Y의 distribution을 N개의 D-dimensional vector로 보았다면, 지금은 D개의 joint-distribution으로 보는 것이다.**
      - $$N(Y; 0, \phi\phi^T + \tau^{-1}I) = \int N(y_d; {\phi}w_d, \tau^{-1}I_{N}) N(w_d; 0, I_{K})dw_d$$
    - 이를 통해 우리는 곧 각 $$y_d$$에 대해 $$W_1$$외에 또 다른 weight을 도입하여 데이터의 GP을 정의할 수 있게 되었다.
    - 이러한 트릭을 통해 Gaussian Process을 우리의 variational parameter $$W1, W2, b$$을 통해 정의할 수 있다.
  - $$p(Y \mid X) = \int p(Y \mid X, W_1, W_2, b)p(W_1)p(W_2)p(b)$$
- 이제 이 variational parameters을 학습해 보자.
  - 우선 mean-field처럼 각 variational parameter $$w_1, w_2, b$$를 따로 구할 것이다.
  - 이때 또 하나의 중요한 트릭이 사용된다.
    - $$q(w) = \prod_{q=1}^{Q}q(w_q)$$에 포함된 하나의 $$w_q$$에 대해 다음과 같은 gaussian mixture로 근사한다.
    - $$q(w_q) = p_{1}N(m_q, \sigma^{2}I_K) + (1-p_1)N(0, \sigma^{2}I_K)$$
    - NN 관점에서 생각하면, 1번 논문처럼 weight을 gaussian으로 해석하긴 하는데, bernoulli $$p_1$$의 확률로 해당 weight을 평균 0으로 끄겠다는 것이다.
      - 여기 $$\sigma$$을 보면 완전히 끄는게 아니라 일정 noise를 추가하는 것으로 보이지만, 이후 절에서 이를 0으로 보내버린다.
    - Dropout을 수식화한 것이라고 생각해도 될 듯 하다.
    - Bias는 그냥 심플하게 dropout을 걸지 않는다.
  - 이제 Variational distribution을 도입했으니 evidence lower bound을 써보자.
    - $$L_{GP} = \int q(W_1, W_2, b)logp(Y \mid X, W_1, W_2, b) - KL(q(W_1, W_2, b) \parallel p(W_1, W_2, b))$$ 
    - 앞서 GP prior p(Y)를 도입했기 때문에 $$logp(Y \mid X, W_1, W_2, b)$$는 쉽게 벗겨진다.
  - 이제 가능한 모든 $$w$$에 대해 integration이 필요한데, 당연히 intractable하다. 
    - 이를 monte-carlo sampling으로 풀텐데, 이때 VAE에서처럼 sampling으로 인한 back-prop 불능의 문제가 발생한다.
  - 따라서 reparameterization trick을 써서 모든 가능한 w가 싹 바뀌진 않고 gaussian noise $$\epsilon$$을 도입해 deterministic part와 stochastic part을 구분한다.
- 이렇게 흘러갈 경우 여타 다른 기법과 차이가 없어진다.
- **여기서는 reparameterization trick을 한 번 더 써서, 앞서 정의한 $$p_1$$에 대한 함수값인 $$z_1$$을 통해 weight를 dropout할 수 있게 한다.**
- 이 경우 이제 sampling을 control할 변수가 $$z_1, \epsilon_1, z_2, \epsilon_2$$로 늘어난다.
- 이때 parameter가 늘어나고 stochasticity가 가중되는 것이 싫기 때문에 $$\epsilon$$을 비트 상에서 정의할 수 있는 가장 작은 수로 근사하여 버린다.
  - 즉 Posterior의 variational distribution을 **large mixture of dirac-delta function**으로 근사한다.
    - **Weight의 모든 statistical variance는 전부 dropout의 bernoulli sampling에서 나온다.**
- 또 한 가지 문제점은 KL 항에서 $$q(w)$$가 gaussian mixture이고, $$p(w)$$는 single gaussian이라는 것인데 이 역시 $$\sigma$$를 굉장히 작게 만들었다는 가정 하에 근사를 통해 analytic하게 구한다.
- 결론적으로 $$L_{GP}$$의 첫 reconstruction 항과 뒤의 KL 항을 합치면 다음과 같다.
  - $$L_{GP} \propto -\frac{\tau}{2}\sum_n \parallel y_n - \hat{y}_n \parallel ^2 - \frac{p_1}{2}\parallel M_1 \parallel ^2 - \frac{p_1}{2}\parallel M_2 \parallel ^2 - \frac{1}{2}\parallel m \parallel ^2$$ (appendix eq.15)
  - 이는 곧 **likelihood에 weight의 variational parameter을 regularize시킨 것과 동일하다.**
  - 분포를 얻었으니 prediction의 uncertainty을 얻을 수 있음은 자명하다.

- 일련의 과정을 다시 요약하면, weight에 dropout을 걸고 stochastic variational inference을 활용함으로써 posterior weight distribution의 학습이 가능함을 보였다.  
  
  
**3. Further Discussion about Dropout**  
- TBD  
