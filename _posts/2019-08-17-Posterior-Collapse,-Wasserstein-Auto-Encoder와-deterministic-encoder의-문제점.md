---
title: Posterior Collapse, Wasserstein Auto Encoder와 deterministic encoder의 문제점
use_math: true
classes: wide
layout: single
---

본 자료는 다음을 참고했습니다.  
- [Wasserstein auto encoder](https://arxiv.org/abs/1711.01558)  
- [On the Latent Space of Wasserstein Auto-Encoders](https://arxiv.org/abs/1802.03761)  
- [reddit discussion](https://www.reddit.com/r/MachineLearning/comments/al0lvl/d_variational_autoencoders_are_not_autoencoders/)
- [Blog post on VAE and posterior collapse, by Paul Rubenstein](http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/)
  
Wasserstein Auto-Encoders(WAE)는 ICLR 2018의 full oral paper로써 기존 VAE의 stochastic encoding에서 오는 문제점을 해결하는 방식을 제안했다. 
Optimal Transport Problem을 도입하여, 기존 VAE의 regularizer 형태를 바꾸고 deterministic encoder를 enable했다는 점에서 많은 주목을 받았다. 
그러나 이후 후속 연구에서 deterministic encoder의 한계점에 대해 서술하며 stochastic encoder에 특정 regularizer를 더한 형식이 가장 유리하다고 
주장했다. 여기서는 우선 기존 VAE 학습 과정에서 나타나는 문제점을 살펴 보고, WAE와 그 후속 연구에 대해 공부하면서 VAE의 장단점에 대해 좀 더 깊이 이해해보자.  
    
  
**-1. VAE and Posterior Collapse**   
Evidence of Lower Bound (ELBO)는 이름을 보더라도, 전개 과정을 보더라도 결국 $$p_{\theta}(x)$$ likelihood을 최대화하는 것이 목적이다.  
이를 Latent Variable Model의 관점에서 조금 다르게 보면, $$KL[p_{data} \parallel p_{\theta}]$$을 좁히는 것이다. 
(잠시 후 보겠지만 Wasserstein Auto-Encoders 역시 이 관점에서 출발한다) 이때 보통은 data가 복잡하고 decoder의 capacity가 한정되어 있기 때문에 이 $$KL$$을 쉽게 좁히기 어렵다. 그러나 만약 decoder가 굉장히 expressive하거나, simple gaussian-like data를 사용한다면 이야기가 달라진다. 굳이 latent variable에 대해 신경쓰지 않아도 optimal latent-independent decoder $$p_{\theta^*}(x \mid z) = p_{data}(x)$$를 찾을 수 있게 된다. 비슷한 원리로 input-independent encoder $$q_{\phi^*}(z \mid x) = p(z)$$를 찾을 수 있다. 즉 encoded latent distribution이 prior distribution과 일치하는, **posterior collapse** 현상이 발생하는 것이다.   
상기한 블로그에서는 이 상황이 곧 VAE의 single global optimum이 될 수 있음을 보였다.   
![image](https://user-images.githubusercontent.com/46081019/63268075-52d12300-c2ce-11e9-9d87-2380ae7c8084.png)  
[출처](http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/)  
  
  
경험적으로 VAE는 이러한 posterior collapse에 취약한 듯 하다. 굳이 data의 complexity 등을 따지지 않아도, 인코더 자체가 stochastic한 분포를 학습해야 하기 때문에 ELBO에서 'reconstruction loss' 자체를 강건하게 줄이기가 어렵다. 또한 강한 regularizer인 prior loss term이 존재하기 때문에 posterior distribution은 곧 prior에 빠르게 collapse되면서 유의미한 정보를 인코딩하는데에 실패하곤 한다. 이로 인해 reconstruction error을 꾸준히 줄이기가 어려워진다. 위의 증명에서 $$q(z \mid x) = p(z), \theta \sim \theta_{subopt}$$인 상황이라고 볼 수 있겠다.
  
WAE는 이러한 상황에 대한 한 가지 해결책으로 each individual posterior distribution이 아니라 aggregated posterior distribution과 prior distribution과의 prior loss을 계산한다. 이를 통해 각 latent distribution이 code 상에서 멀어지는 것을 꾀한다.  
  
  
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
  - 이 중요한 information 항은 regularizer에서 분리시키면서 더 좋은 representation learning을 할 수 있다.
  - 또한 다음 페이퍼에서 이 항의 역할을 한 문장으로 잘 설명하고 있다.
  - > To better understand the difference, note that $$Q_Z$$ is the distribution obtained by averaging conditional
distributions $$Q(Z \mid X = x)$$ for all different points $$x$$ drawn from the data distribution $$P_X$$. **This means that WAEs explicitly control the shape of the entire encoded dataset while VAEs constrain every input point separately.**
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
- WAE의 1저자인 Ilya Tolstikhin이 참여한 후속 연구이다.
- 바로 실험 결과를 확인해 보자.
- ![image](https://user-images.githubusercontent.com/46081019/63206511-8b1b0a80-c0f0-11e9-9f21-6f9ddbf2a6c7.png)  
  - Deterministic encoder와 Random encoder을 비교하는 figure이다.
  - 데이터는 manifold의 차원($$dim_I$$) = 1인 간단한 toy 데이터이다.
  - 현재 latent z의 차원 = 2로, $$dim_Z >> dim_I$$인 상황이다. 
  - 이때 deterministic encoder을 이용해 two-dimensional latent space 위에 one-dimensional data을 코딩하면, 2차원 상에서 이상하게 꼬인 하나의 non-linear entity가 나온다. (Up-middle)
  - 반대로 stochastic encoder를 이용해 전체 latent space 위에 확률적으로 data를 코딩하면, latent variable $$z$$의 평균은 one-dimensional data를 $$z2$$ 축을 따라 잘 설명하게 된다. 이때 나머지 한 $$z1$$ 축은 random noise로 채워진다.
  - 즉 encoder가 모든 latent space를 활용하여 z를 코딩하기 때문에, decoder가 latent space의 'hole'을 신경쓰지 않고 reconstruction을 할 수 있다.
  - 반면 deterministic encoder를 쓴 경우 latent space의 unknown-region이 많아지기 때문에 decoder가 제대로 학습되기 어렵다.
- 이때 만약 stochastic encoder를 쓰더라도, latent dimension이 지나치게 크면 전체 data에 대해서 제대로 dimension reduction이 이뤄지기 어렵고, 특정 latent variable이 그냥 단일 constant로 collapse해버리는 상황이 벌어진다고 한다.
  - 이는 곧 stochasticity가 의미 없어지고 deterministic encoder로 수렴하는 상황을 의미한다.
- 따라서 논문에서는 이를 막기 위해 latent space의 각 dimension에 대해 logarithm of encoded variance를 regularizer로 넣어줘서 encoded variance가 1로 되게끔 만들었다고 한다.
