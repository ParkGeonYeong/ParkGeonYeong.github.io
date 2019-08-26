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
- 기존의 많은 NN은 대부분 deterministic layer으로 구성되어 있다.
  - 반면 weight를 "분포"화하면, weight의 sampling에 따라 ensemble of networks를 학습시킬 수 있다.
    - 여기까지 보면 어떤 면에서 dropout과 비슷하다.
    - 실제로 그 다음 논문은 weight의 posterior 분포를 dropout으로 근사한다.
  - 이를 통해 data sample에 대한 uncertainty을 quantify할 수 있고, overfitting을 피할 수 있다. 
  - 여기서는 NN의 앙상블을 학습시키면서도 weight를 mean과 diagonal covariance $$\mu, \rho$$으로 근사하기 때문에, 파라미터가 크게 늘어나지 않는다.
    - 그 다음 논문에서는 이것도 많다며 아예 추가 파라미터를 도입하지 않는다.
- 이렇게 weight의 분포를 잡으면 문제는 MLE에서 MAP으로 바뀐다.
  - $$W^{MLE} = argmax_{w} logP(D \mid w)$$
  - $$W^{MAP} = argmax_{w} logP(w \mid D) = argmax_{w} logP(D \mid w) + logP(w)$$
  
  
**2. Dropout as Bayesian Approximation**  
- **핵심 아이디어: Bernoulli dropout만으로도 Gaussian Process의 integration over weight space을 근사할 수 있다.** 
  - NN의 W1, W2, b를 이용해 PSD kernel 정의 
  - Kernel을 통해 p(y|x, w)을 GP로 만들 수 있음을 보임
  - 이때 Intractable True Weight Posterior를 ELBO로 학습함
  - 이 과정에서 dropout과 reparameterization trick(Kingma et al.,)을 통해 ELBO의 계산량을 확 줄임
  - 정확히는 Monte-Carlo dropout만으로도 ELBO를 얻을 수 있게 함!
  - 결국은 reparameterization trick을 두 번 쓴 셈
  - 이를 통해 RL 실험에서는 posterior sampling을 근사하기도 함   
  
  
**3. Further Discussion about Dropout**  
- TBD  
