---
title: Variational Auto Encoder(VAE)
use_math: true
classes: single
layout: wide
---

*이전 포스트와 이어집니다.*
- [Gaussian Process와 Variational Inference](https://parkgeonyeong.github.io/Gaussian-Process%EC%99%80-Variational-Inference/)   

*다음 자료를 주로 참고했습니다.*  
- [Original Paper](https://arxiv.org/pdf/1312.6114.pdf)
- [Jaejun Yoo님 블로그 포스트](http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html)  

출판 이후 생성 모델로 꾸준히 활용되고 있는 VAE이다. 
Variational inference, VAE, bayesian neural network 등은 모두 풀고자 하는 목표가 동일하다. 
이전 포스트에서 Variational Inference lower bound, 혹은 ELBO의 유도는 log likelihood $$logp(x)$$의 최대화(MLE)에서 시작했다. 
'Auto-encoder'라는 이름에서 알 수 있듯이, VAE는 self-supervised MLE 문제를 푸는 것이며, 
bayesian neural network는 보다 일반적인 training set에 대해 MLE를 푼다고 할 수 있다. 
여기서는 VAE의 의의, 수식적 전개와 모델 구조를 주로 알아보겠다. 

**0. Introduction**  
Variational Inference의 목표를 간단히 정리하면 'Intractable한 posterior distribution을, 
이미 알고 있는 variational distribution으로 근사하는 것'이라 할 수 있다. 
VAE에서는 given data x에 대한 latent variable z의 분포, $$p(z \mid x)$$가 posterior distribution이다. 
이때 given x를 잘 represent하는, true z가 정해져 있지 않다는 문제가 생긴다. 가령 우리가 true prior distribution $$p_\theta(z)$$를 잘 알고 있다면, 
$$z$$을 sampling하고 이에 대응하는 $$x$$를 확률적으로 생성할 수 있을 것이다. ($$p_\theta(x \mid z))  
하지만 현재 우리는 generative model $$\theta$$, latent variable $$z$$을 모두 모르기 때문에 x를 reconstruct하기 쉽지 않다. 
$$z$$를 아예 모르기 때문에 기존 variational inference에서 많이 활용되어 온 mean-field 방식도 사용할 수 없다. (모든 가능한 $$z_i$$에 대해 integration이 필요하기 때문, 이전 포스트의 1.1 Factorized distribution 참고)  
굉장히 많은 x에 대해 가능한 z를 모두 sampling하고 이를 validation하는 MCMC도 computational load가 높기 때문에 활용이 어렵다. 
([MCMC와 posterior sampling](https://parkgeonyeong.github.io/Markov-Chain-Monte-Carlo%EC%99%80-Posterior-Sampling/) 참고)  
  
**따라서 VAE에서는 주어진 x로부터 $$z$$를 "encoding"하는 variational approximation $$\phi$$와, 
이로부터 다시 x를 생성하는 generative model $$\theta$$ 두가지 모두 end-to-end로 학습하여 문제를 해결한다.**
과정은 아래 그림과 같다. Coding의 관점에서 q는 encoder, p는 decoder로 불린다.   
![image](https://user-images.githubusercontent.com/46081019/57665961-18a21680-7639-11e9-8391-154165db5abb.png)  
    
**1. Variational Bound and reparameterization trick**  
$$lnP(x) = \int{q(w)ln\frac{P(X \mid w)p(w)}{q(w)}dw}+KL(q \mid\mid p(w \mid X))$$를 유도했었다. 
이때 KL-divergence는 non-negative하므로 우변의 첫 항은 MLE의 lower bound, ELBO가 된다. 이 ELBO를 $$L$$로 표현하여 논문의 꼴로 쓰면 다음과 같다.  
$$logP_\theta(x^i) \geq L(\theta, \phi, x^i) = E_{q_\phi(z \mid x)}[-log{q_\phi(z \mid x)} + log{p_\theta(x, z)}]$$_ 
$$p_\theta(x,z)$$를 분리하여 다시 쓰면 다음과 같다.   
$$L(\theta, \phi, x^i) = -D_{KL}(q_\phi(z \mid x_i) \mid\mid p_\theta(z)) + E_{q_\phi(z \mid x)}[log{p_\theta}(x^i \mid z)]$$  
  
(KL-div에 negative가 붙음)
$$q_\phi, p_\theta$$가 어떤 형태인지는 조금 나중에 알아보고, 식 자체에만 집중해 보자.  
Lower bound $$L$$을 최대화하기 위해서는 $$\phi$$에 대해 gradient를 구해서 $$\phi$$를 최적화해야 한다. 
이때 KL divergence는 analytic한 해가 존재한다. (나중에 이를 보인다) 하지만 $$L$$에 포함되어 있는 expectation항은 analytical하게 구하기가 매우 어렵다. 
따라서 가장 처음 생각해 볼 수 있는 방식은 분포 q를 따르는 z을 *monte-carlo sampling*하여 expectation 항의 conditional likelihood를 경험적으로 구하는 것이다. 하지만 이렇게 될 경우 random sampling operation이 들어간 채 back propagation을 해야 하므로 indifferentiable 상태가 된다.  
**이를 굉장히 스마트하게 해결한 방식이 바로 reparameterization trick이며, 논문의 핵심이라고 볼 수 있다.**  
Reparameterization이 무엇인지 보기 전에, 우선 최종적으로 구한 loss를 보자.   
![image](https://user-images.githubusercontent.com/46081019/57667711-90733f80-763f-11e9-9e16-ae1d612da463.png)  
식 자체는 L개의 latent variable z를 sampling하고, 이를 통해 loss를 구한 것으로 monte-carlo sampling과 큰 차이가 없어 보인다. 
하지만 $$z$$의 식이 변한 것을 알 수 있는데, reparameterization function $$g_\phi$$, 그리고 어떤 random variable $$\epsilon$$이 등장하였다. 
    
Reparameterization는 말 그대로 우리의 관심 대상 parameter를 다시 잡는다는 뜻이다. 
즉 우리의 관심사인 $$z$$를 $$q$$에서 바로 sampling하면 미분이 불가능하기 때문에, $$\epsilon$$를 대신 sampling하여 $$z$$를 indirect하게 구하는 것이다. 이를 통해 우리는 적어도 $$z$$에 대해서 gradient를 전달할 수 있게 된다. 
그림으로 표현하면 다음과 같다.  
![image](https://user-images.githubusercontent.com/46081019/57667976-a2a1ad80-7640-11e9-8b08-588e2870c130.png)  
예시를 들어 Reparmeterization function g를 알아보자. 
만약 우리가 latent variable $$z$$를 multi-variate distribution으로 가정하고, 
$$\phi$$는 이 multi-variate gaussian의 mean과 covariance를 의미한다고 하자. 
즉 $$Q(Z \mid X)=N(\mu(X),\sum(X))$$일 때, $$g(\epsilon, \phi, x) = \mu(X) + {\sum}^{\frac{1}{2}}(X)*\epsilon$$이라고 할 수 있을 것이다.   
식을 보면 g는 $$\phi$$에 대해 미분 가능하며, 따라서 latent variable z의 underlying distribution parameter $$\phi$$를 최적화할 수 있다. 
이때 $$\epsilon$$를 sampling하는 방식은 함수 $$p$$의 선택에 따라 다양하게 가능하다.  
  
**2. Model architecture**  
VAE의 전체 model 구조를 보면 다음과 같다.  
![image](https://user-images.githubusercontent.com/46081019/57668350-011b5b80-7642-11e9-9c12-1ec079bdd8f7.png)  
왼쪽은 monte-carlo sampling 버전, 오른쪽은 reparameterized 버전이다. 파란색 박스가 합쳐져서 우리가 최대화하고자 하는 최종 loss가 된다.  
하나씩 성분을 살펴보면 다음과 같다. 다시 한 번 강조하면, 모든 항들은 MLE problem으로부터 ELBO를 유도하는 과정에서 나왔으며 
따라서 ELBO를 이해해야 VAE의 loss를 이해할 수 있다.
- Encoder
  - 목적 : $$\phi$$ (여기서는 평균과 공분산)을 encoding한다. 이를 통해 latent variable를 construct할 수 있다.
  - 모델 : MLP를 사용한다.
- KL divergence term of ELBO
  - 목적 : **ELBO에서 negative KL-divergence는 일종의 'regularizer'같은 역할을 한다.** 
  Latent variable의 prior distribution을 $$p(z;0, I)$$으로 잡을 때 $$\phi$$가 여기서 너무 멀어지지 않도록 한다. 
