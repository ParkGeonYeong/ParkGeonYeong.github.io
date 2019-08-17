---
title: Wasserstein Auto Encoder와 deterministic encoder의 문제점
use_math: true
classes: wide
layout: single
---

본 자료는 다음을 참고했습니다.  
- [Wasserstein auto encoder](https://arxiv.org/abs/1711.01558)  
- [On the Latent Space of Wasserstein Auto-Encoders](https://arxiv.org/abs/1802.03761)  

  
Wasserstein Auto-Encoders(WAE)는 ICLR 2018의 full oral paper로써 기존 VAE의 stochastic encoding에서 오는 문제점을 해결하는 방식을 제안했다. 
Optimal Transport Problem을 도입하여, 기존 VAE의 regularizer 형태를 바꾸고 deterministic encoder를 enable했다는 점에서 많은 주목을 받았다. 
그러나 이후 후속 연구에서 deterministic encoder의 한계점에 대해 서술하며 stochastic encoder에 특정 regularizer를 더한 형식이 가장 유리하다고 
주장했다. 일련의 과정을 공부하면서 VAE의 장단점에 대해 좀 더 깊이 이해해보자.  
  
  
**0. Wasserstein Auto-Encoders**  
  
- 이름에 Wasserstein이 붙은 이유는 Generative Distribution $$P_G$$와 Real Distribution $$P_X$$의 wasserstein distance을 minimize하기 위해서이다. 
- 안타깝게도 대부분의 divergence metric으로는 unknown distribution $$P_X$$와, NN으로 parameterize된 $$P_G$$의 거리를 측정하기 어렵다. 
- 이를 해결하기 위해 많은 generative model이 KL-divergence(VAE, equivalently cross entropy), f-divergence, Wasserstein distance 등을 활용했다.
- WAE에서는 WGAN처럼 wasserstein distance를 줄이는 것을 목표로 하는데, 차이점은 WAE는 latent space를 이용하며 이를 얻을 수 있다는 점이다.
  - 이를 통해 VAE의 장점은 유지하면서, WGAN처럼 좋은 resolution image을 얻을 수 있다.

- $$P_G$$와 $$P_X$$의 거리를 Kantorovich의 original formulation으로 나타내면 다음과 같다. 
  - $$W_C(P_X, P_G) = inf_{\Gamma \in P(X \sim P_X, Y \sim P_G)} E_{(x,y) \sim \Gamma}[c(x, y)]$$  
    - [WGAN 이전 포스트 참고](https://parkgeonyeong.github.io/GAN-%EC%A0%95%EB%A6%AC/)
    - 참고로 WGAN, WGAN-GP에서는 위 식을 Kantorovich-Rubinstein의 dual problem으로 전환하여 풀어낸다.
    - 결국 joint distribution $$\Gamma$$를 찾기 어려워서인데, WAE에서는 kantorovich의 dual problem 말고 다른 방식을 사용한다. 
  - 아래 Thm 1이 논문의 핵심이다.  
  - ![image](https://user-images.githubusercontent.com/46081019/63206031-72a6f200-c0e8-11e9-8b30-0a6dc7110d23.png)  
  - 식을 보면 X와 Y의 모든 조합에 대해서 최적화하는 것이 아니라, 인코더를 통해 얻어낸 $$Q(Z \mid X)$$ 상에서 최적화를 한다. 
    - 따라서 언급했다시피, Wasserstein distance를 사용하지만 꼴은 VAE와 유사하다.
    - 그러나 Q의 제약 조건이 $$Q_z = P_z$$임을 특히 기억하자. 
  - 이 제약 조건을 라그랑지안 제약으로 사용하여 Objective function을 정의하면 다음과 같다. 
  - ![image](https://user-images.githubusercontent.com/46081019/63206068-e9dc8600-c0e8-11e9-8a24-8e48c61a22eb.png)
  - VAE 식과 직접 비교해보자. 
  - ![57672305-768f2800-7652-11e9-8ba3-3cea071c1dd4](https://user-images.githubusercontent.com/46081019/63206117-830b9c80-c0e9-11e9-86fd-0017c0478a4d.png)  
  - Regularizer 항을 비교하면, VAE의 경우 모든 $$x_i$$에 대해 $$Q(z \mid x_i)$$와 $$P(z)$$ prior 간의 KL-divergence으로 정의되어 있는 한편, 
  WAE의 경우 모든 $$x_i$$에 대해 aggregated된 하나의 $$Q_z$$가 regularized되고 있다. 

- 이 regularizer는 사실 보기보다 큰 차이이다.
  - [Disentagling and VAE 이전 포스트 참고 (작성중)](https://parkgeonyeong.github.io/VAE%EC%99%80-Disentanglement/)
  - [ELBO surgery 참고](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf)
  - 요약하자면 기존 VAE regularizer는 $$KL(q(z) \parallel p(z))$$ 외에 $$x_i$$와 $$z$$의 mutual information을 추가로 포함하는 항이다. 
  - 이 information을 분리시키면서 더 좋은 representation learning을 할 수 있다.
- ![image](https://user-images.githubusercontent.com/46081019/63205693-18576280-c0e3-11e9-869c-1579594a197b.png)  
- 위의 figure가 WAE의 장점을 잘 요약하고 있다.  
- VAE는 *each latent distribution*이 prior에 정규화되지만, WAE는 *aggregated latent distribution*이 prior에 정규화된다.
  - VAE의 경우 모든 latent distribution이 동일한 global prior에 가까워지면서 distribution 간의 overlapping이 생길 수 있고, 
  이는 곧 분포가 서로 몰려 reconstruction이 collapse될 수 있음을 의미한다. 
  - 반면 WAE는 $$Q_z$$에 대해 정의되었기 때문에 굳이 stochastic encoder를 사용할 필요가 없다.
  - 기존 VAE는 stochastic encoding으로 인해 z의 randomness가 발생했고, 이 random한 z을 통해 제대로 x'을 reconstruct하는게 어려웠다.  
  따라서 WAE가 deterministic encoder을 사용할 수 있다는 점은 굉장히 큰 변화를 시사한다. 
  (하지만 바로 다음 페이퍼에서 곧 stochastic encoder가 더 필요함을 보였다...)
- 아무튼 WAE에서 정규화항이 일반적인 divergence metric에 대해 정의되어 있기 때문에, 두 가지 loss를 활용해 실험을 진행한다.
  - WAE-GAN은 divergence metric을 JS-divergence로 잡고 GAN을 통해 풀어낸다.
    - 참고로 일반적인 GAN처럼 unknown complex data distribution이 아니라 simple unimodal gaussian $$P_Z$$에 대한 JS-div이기 때문에 학습이 더 쉽다고 한다.
    - 그러나 어쨌든 min-max 방식이기 때문에 나중에는 2번째 방식인 WAE-MMD을 더 많이 쓴다.
  - WAE-MMD는 maximum-mean-discrepancy라는 metric을 잡는다.
  - 결국 두 방식 모두 deterministic encoder로 가능하기 때문에 잡은 것 같다. 
  - 물론 KL-divergence로 stochastic하게 해도 상관은 없을 것이다.
- 실험 결과는 생략한다. 

**1. On the Latent Space of Wasserstein Auto-encoders**  
