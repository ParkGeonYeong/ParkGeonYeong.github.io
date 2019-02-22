---
title: "Implicit Quantile Networks 리뷰 (Distributional RL 일대기)"
use_math: true
layout: single
classes: wide
---

본 자료는 다음을 주로 참고했습니다.
- [RL Korea DIST_RL 시리즈](https://reinforcement-learning-kr.github.io/2018/09/27/Distributional_intro/)
- [IQN RL](https://arxiv.org/abs/1806.06923)

**0. Distributional RL이란**  
Distributional RL은 에이전트의 학습 시 자연적으로 발생하는 intrinsic randomness를 고려해, reward를 random variable로 풀어내는 알고리즘입니다. 
에이전트가 놓인 다양한 환경에 따라 리워드가 높은 분산의 분포를 따를 수도 있고, multi-modal distribution을 띌 수도 있습니다. 
기존 value-based RL 알고리즘이 이러한 리워드 분포를 single estimated Q value(기댓값)으로 대체하여 접근했던 것과 다르게, 
distributional RL은 리워드 분포 자체를 학습 타겟으로 삼습니다. 딥마인드의 Will Dabney를 필두로 저자들이 주장하는 장점은 다음과 같습니다.
- Reduced Chattering  
  Bellman Optimality Operator는 수학적으로 수렴이 보장된 알고리즘이지만, (by gamma-contraction)
  학습 과정이 불안정하다는 한계점이 있었습니다. (A Distributional Perspective on Reinforcement Learning) 
  이로 인해 point approximation 과정에서 수렴하지 않는 chattering 현상이 발생하는데, distribution을 예측하는 과정에서 이를 줄일 수 있다고 합니다.
- State aliasing
  Pixel state 자체는 매우 비슷하지만 예상되는 결과가 매우 다른 경우에 대해서 distribution을 통해 구분 가능 (Intrinsic stochasticity)
- Richer set of prediction  
  
2017년 발표된 C51이라는 이름의 알고리즘을 시작으로 QR-DQN, IQN까지 3가지 시리즈의 논문이 딥마인드에서 발표되었습니다. 
C51의 경우 reward distribution의 supports 개수와 최대,최소값을 파라미터로 지정했습니다. 
따라서 state에 대한 네트워크의 output dimension은 [support의 수 * action의 수]가 되었습니다. 지정해야 하는 hyper-parameter가 너무 많았고 
이는 다양한 task에 대한 robustness를 약화시킬 위험이 있었습니다. 
또한 더 큰 문제로는 true distribution을 향한 수학적인 수렴성이 보장되지 않는다는 것이였는데, 
이는 뒤에서 다루겠지만 C51에서 사용한 loss가 cross-entropy였기 때문입니다.  이 외에도 복잡한 projection 등 다양한 문제가 있었고, 
이 문제들을 개선한 논문이 바로 QR-DQN입니다. 그 후속으로 이어서 IQN이 나왔고 현재까지 다양한 아타리 게임에서 SOTA를 기록하고 있는 알고리즘입니다. 
그 기초가 된 C51 역시 매우 중요한 논문이지만, 본 자료에서는 이 QR-DQN과 IQN에 대해 주로 다루겠습니다.  

**1. Quantile Regression DQN과 그 한계점**  

**2. IQN RL**
