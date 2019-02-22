---
title: "Implicit Quantile Networks 리뷰 (Distributional RL 일대기)"
use_math: true
layout: single
classes: wide
---

본 자료는 다음을 주로 참고했습니다.
- [RL Korea DIST_RL 시리즈](https://reinforcement-learning-kr.github.io/2018/09/27/Distributional_intro/)
- [IQN RL](https://arxiv.org/abs/1806.06923)
- [Wasserstein metric](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)

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
그 기초가 된 C51 역시 매우 중요한 논문이지만, 본 자료에서는 이 QR-DQN, 더 중요하게는 IQN에 대해 다루겠습니다. C51에 대해서는 논문 및 RL Korea 블로그를 미리 참고 바랍니다.  
  
**1. Quantile Regression DQN**   
우선 Wasserstein metric에 대해 알아보겠습니다.   
Wasserstein metric은 분포 공간에 대한 거리 metric으로, 직관적으로는 두 확률 분포의 떨어진 거리를 의미합니다. 
이는 optimal transport 문제로도 잘 알려져 있습니다. 분포 $${\mu}(x)$$에서 $${\pi}(x)$$으로 값을 '옮긴다'고 생각해 봅시다.
$$\mu$$의 x에서 $$\pi$$의 y로 값을 옮길 때, 받아 얼마나 많은 '값을 옮길지'를 리턴하는 joint 함수 $${\gamma}(x,y)$$가 있고, 두 지점의 거리에 대한 함수 $$c(x,y)$$가 있습니다. 이때 미소 $$dx, dy$$에 대해서 모든 거리 비용을 계산하면   
$$\int \int c(x,y)\gamma(x,y)dxdy = \int c(x,y)d\gamma(x,y)$$  
가 됩니다. 이때 $$\mu, \pi$$의 모든 가능한 결합 확률 분포(joint distribution)에 대해서 거리 비용 기댓값의 '가장 작은' 추정값은 다음과 같습니다.  
$$\inf_{\gamma\in\Gamma(\mu,\pi)}\int c(x,y)d\gamma(x,y)$$  
이때 $$\Gamma$$는 $$\mu, \pi$$의 sampling된 분포 사이에서 취할 수 있는 marginal joint distribution의 집합입니다. 이를 wasserstein metric이라고 정의합니다. Intuitive한 의미 정도만 이해되고 수학적으로 엄밀하게는 해석학적 지식이 필요한 것 같습니다. 
또한 논문에서 더 많이 사용하는 형식은 inverse CDF function 또는 quantile function을 활용한 식으로 다음과 같습니다. (p-Wasserstein)  
$$W_p(U,V) = \left(\int_{0}^{1}ㅣF_U^{-1}(w)-F_V^{-1}(w)ㅣ^pdw\right)^{1/p}$$  
직관적으로 이해하기로는 동일한 percentile값(함수의 y) $$w$$에 대해서 두 quantile function의 $$L^p$$ 거리 metric을 잰 다음, 
이를 y축의 값에 대해서 적분한 것으로 받아들였습니다. 혹시 정확한 의미를 설명해 주실 수 있는 분은 댓글로 해주신다면 감사드리겠습니다!  
  
Quantile Regression 이전의 C51 논문에서는 Bellman optimality operator에 대해 wasserstein metric이 gamma-contraction으로 수렴한다는 점을 밝혔습니다. 즉 loss로 이 metric을 쓴다면 C51 알고리즘이 수학적으로 value distribution을 ground truth로 수렴시킬 수 있다는 것입니다. 하지만 해당 논문에서는 wasserstein metric은 SGD로는 optimal하게 minimum으로 감소시킬 수 없다는 점도 밝혔습니다. 따라서 C51이 아닌 cross-entropy를 이용해 두 분포의 loss를 정의했고 이는 곧 C51이 수학적으로 불완전한 알고리즘임을 의미합니다. (구체적인 증명은 확인하지 못한 상태로 C51 논문을 참고 바랍니다)  

**QR-DQN은 C51의 불완전한 수렴성을 보완한 알고리즘입니다.** 논문에서 QR-DQN은 C51과 'transpose' 관계라고 설명하고 있습니다. 이는 C51은 supports에 대한 확률 분포를 estimate하는 반면 QR-DQN은 $$1/N dirac$$의 고정된 확률 분포 아래 N개의 supports을 estimate하기 때문입니다. 즉 N개의 quantile을 구하고 있습니다. 이를 갖고 처음 할 수 있는 시도는, 앞서 C51에서 원래 하려 했던 것처럼, 이 quantile 분포를 파라미터화해서 target quantile 분포 간의 Wasserstein metric을 직접적으로 줄이는 것입니다. 하지만 아쉽게도 이 역시 SGD를 바로 적용시킬 수 없다는 문제가 있습니다.

하지만 C51과 달리 QR-DQN에서는 quantile function을 갖고 있기 때문에 quantile regression을 시도를 할 수 있습니다. 이는 **quantile function을 unbiased하게 approximate**하는 알고리즘으로 경제학에서 많이 쓰인다고 합니다. 네트워크를 통해 초기 결과로 얻은 value distribution의 quantiles와, bellman operator를 통해 얻은 target quantiles에 quantile regression을 적용시키는 것입니다. **즉 두 quantile function의 quantile regression loss(asymmetric convex loss)를 SGD를 통해 minimize할 수 있다는 것이 QR-DQN의 핵심입니다. 이것이 궁극적으로는 inverse CDF, quantile function의 metric을 최소화함으로써 Wasserstein metric의 최소화가 가능하다는 것입니다.** 구체적으로는 주어진 value distribution에 bellman optimal operator를 적용시킨 $${\tau}^{\pi}z$$을 타겟으로 하여, quantile regression을 적용해 타겟에 맞는 approximated distribution $$Z$$을 얻었다고 합시다. 이때 $$Z$$은 p-wasserstein metric에 대해 contract하기 때문에 unique fixed point $$Z^{\pi}$$로 수렴을 보장받게 된다는 것이 전체 증명입니다.  

**2. IQN RL**  
QR-DQN에서 한 발 더 나아가, IQN은 N개의 *discrete and fixed* quantiles이 아닌 전체 continuous한 quantiles function을 estimate하게 됩니다. QR-DQN이 고정적으로 Quantiles 개수와 대응되는 확률($$\tau$$)을 고정하고 운영했다면 IQN은 이를 random하게 sampling합니다. 이를 $$\tau-sampling$$이라고 합니다. 얼핏 quantile function을 non-uniform하게 estimate한다는 점을 빼곤 동일하지만, 변화를 통해서 얻을 수 있는 가장 큰 장점으로는 시도할 수 있는 class of policy가 다양해진다는 것입니다. 구체적으로는 e-greedy policy 외에도 **risk-sensitive policies**을 시도할 수 있습니다. (risk-sensitive policy에 대한 내용은 마찬가지로 RL Korea 블로그 참고) 여기서 말하는 risk는 return의 distribution에서 생기는 intrinsic uncertainty를 말합니다. 분산이 크지만(매우 낮은 value가 나올 수도 있지만) 평균은 높은 value를 sampling할 것인지, 분산이 작아 어느 정도는 값이 보장되지만 평균이 낮은 value을 samplling할 지에 대한 policy라고 생각할 수 있습니다.  
  
$$\tau-sampling$$이 어떻게 risk-sensitive policy을 가능하게 하는지 알아보겠습니다. 기존 distributional RL에서도 액션을 선택할 때 Q-value를 구해 사용합니다. 이때 Q-value는 supports value의 확률 분포 $$z$$을 이용한 기댓값입니다. 이를 조금 복잡한 수식으로 나타내면 다음과 같습니다.  
$$Q = E[U(z)]$$   
이를 expected utility theory라고 하며 U를 utility function이라 합니다. 만약 U가 identity function인 경우 앞서 서술한 과정과 동일합니다. 
하지만 U가 'linear'하지 않고 concave하거나 convex한 non-linear function인 경우 policy는 다른 경향을 나타내게 됩니다. 
구체적으로는 U가 linear한 경우 risk-neutral, concave한 경우 risk-averse, convex한 경우 risk-seeking policy라고 합니다. U가 극단적으로 convex한 경우를 예로 생각해 보겠습니다. 임의의 z가 convex의 minimum에서 매우 먼 경우, utility function은 매우 큰 값을 리턴합니다. 즉 높은 z value를 seeking한다고 할 수 있습니다. 확률이 100%인 1과, 30%로 2, 70%로 0.1인 경우를 생각해보면 convex utility function은 낮은 확률의 2라도 매우 큰 값으로 가치를 뻥튀기하기 때문에 후자를 선택할 확률이 높아지게 됩니다.   
현재 우리는 distribution $$z$$을 직접적으로 갖고 있지 않고 quantile function을 갖고 있기 때문에, 각 quantile에 해당하는 확률 $$\tau$$를 샘플링해 이에 해당하는 value $$z_{\tau}$$을 얻어야 합니다. 이를 함수로 표현하면 $$Z_{\tau}=F_z^{-1}(\tau)$$, $$\tau=U([0,1])$$이라 할 수 있습니다.  
  
이제 $$\tau$$를 sampling하는 logic에 대한 함수를 $$\beta(\tau)$$라 합시다. 위의 Q-value 식을 다시 표현하면 다음과 같습니다.  
$$Q_{\beta}(s,a) = E[Z_{\beta(\tau)}(s,a)]$$  
이때 만약 $$\beta$$가 $$\tau$$를 매우 치우친 값에서 sampling하도록 유도한다고 가정합시다. 가령 [0, 1] 사이의 $$\tau$$를 받아 [0, .25] 사이의 값을 return한다면, 그 결과인 value 역시 최대 quarter-quantiles을 넘지 못할 것입니다. 즉 액션별로 하위 25%의 return을 비교해서 어떤 액션이 안정적으로 return을 보장하는지를 확인하는 것입니다. **따라서 expected return이 높진 않더라도 non-risky한 action을 선호하게 되는데, 이를 risk-averse policy라 합니다.** 반대로 어떤 action이 Expected return은 높아도 Variance가 커서 매우 낮은 return을 받을 가능성이 있다면, 선택받지 못할 것입니다. **이를 distorted expectation of $$Z(s,a)$$라고 하며, $$\beta$$는 distortion risk measure라고 합니다.** 어떤 $$\beta$$를 선택하는 지가 policy의 성격을 결정한다고 할 수 있겠습니다.  
  
  
지금까지 C51, QR-DQN, IQN 등으로 불리는 알고리즘 각각의 가장 큰 특징에 대해 다뤄보았습니다. IQN은 atari에 다양한 policy를 적용하면서 task 특성에 맞는 policy를 찾으려는 모습을 보여줬고, 성능 역시 가장 좋았습니다. 딥마인드에서 앞으로 distributional RL 알고리즘을 어떤 방향으로 발전시킬지 기대됩니다.



