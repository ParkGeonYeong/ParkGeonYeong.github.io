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
  - **이후 이 Intractable True Weight Posterior를 ELBO로 학습한다**.
  - 이 과정에서 dropout과 reparameterization trick(Kingma et al.,)을 통해 ELBO의 계산량을 확 줄인다.
  - 정확히는 Monte-Carlo dropout만으로도 ELBO를 얻을 수 있게 함!
  - 결국은 reparameterization trick을 두 번 쓴 셈이다.
  - 이를 통해 RL 실험에서는 posterior sampling을 근사하기도 함
  - 앞선 논문과의 차이점은, weight uncertainty를 mu와 stddev으로 parameterize하느냐, monte carlo sampling으로 parameterize하느냐의 차이이다.
    - Dropout을 도입하여 Gaussian Process의 weight space of view을 Neural Network 차원에서 해석한다.
- 
  
  
**3. Further Discussion about Dropout**  
- TBD  
