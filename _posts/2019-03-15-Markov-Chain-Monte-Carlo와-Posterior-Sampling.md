---
Title : "Markov Chain Monte Carlo와 Posterior Sampling"
use_math : true
---

**0. Posterior Sampling의 필요성**  
Bayes Inference에서 posterior distribution을 정확히 구하는 것은 계산 측면에서 매우 어려운 일이다. 
$$P(\thetaㅣx) = \frac{P(xㅣ\theta)P(\theta)}{P(x)}$$에서 분모의 evidence $$P(x)$$를 marginalization 과정으로 계산해야하기 때문이다.
모든 가능한 $$\theta$$에 대해 이를 계산하는 것은 practical하게는 불가능한 일이다. Analytically intractable한 경우가 많다. 

따라서 우리가 원하는 posterior distribution을 sampling 과정으로 근사해서 구하는 과정(approximate inference)을 생각해 볼 수 있는데, 
많이 쓰이는 알고리즘이 바로 Markov Chain Monte Carlo(MCMC)이다. 여기서는 우선 몇몇 sampling 기법을 먼저 다룬 다음, 
해당 과정들의 문제점을 보완할 수 있는 MCMC를 설명하겠다.  

**1. 다양한 Sampling Methods**  

**1.1 Standard Sampling**  ion

**1.2 Rejection Sampling**  
**1.3 Importance Sampling**  
**2. Markov Chain Monte Carlo**  

MCMC는 앞선 Sampling 알고리즘이 고차원 분포에서 좋지 못한 성능을 보이는 것을 개선할 수 있는 알고리즘이다.  
  
가령 rejection sampling에서 제안 분포 $$q(z)$$을 먼저 결정할 때, 우리 관심사인 $$p(z)$$와 매우 유사하도록 만들어야 승인률이 올라간다. 
예를 들어 $$p(x)$$가 zero-mean D-dimensional 가우시안 분포인 경우를 생각하자. 굉장히 이상적으로 $$q(x)$$를 동일한 zero-mean 가우시안으로 설정하면, 
$$\sigma_q^2 > \sigma_p^2$$ 조건을 만족해야만 $$q(x)$$가 rejection sampling에서 사용될 수 있다.  
이때 k는 $$k=({\sigma_q}/{\sigma_p})^D$$이 된다고 한다. 즉 매우 파라미터가 고차원인 분포의 경우, 
k는 기하급수적으로 올라가며 따라서 승인률은 매우 낮아질 수 밖에 없다. 결국 샘플링의 효율은 너무 떨어지며, 이는 Importance Sampling에서도 비슷하다. 
  
이를 해결하기 위한 방식이 MCMC이다. Markovian 특성, Monte Carlo 등은 따로 설명하지 않고 알고리즘의 특성과 증명에 집중하겠다. 
굉장히 복잡하거나 사전 지식이 없는 분포 $$p(z)$$를 대상으로, 앞서 rejection sampling과 importance sampling 때처럼 
제안 분포 $$q(z)$$을 설정한다. 이 $$q$$를 이용하여 $$z^(\tau)$$를 sampling하게 되는데, 
이를 markov chain에서의 state at time $$\tau$$로 사용하는게 알고리즘의 시작이다. 
해당 state에서 다시 분포 q를 이용해 새로운 state $$z^*$$를 샘플링했다고 하자. 
이때 MCMC에는 rejection sampling과 유사하게, 이 새로운 state의 인정 여부를 결정하는 승인 과정을 거친다. 
알고리즘마다 다르지만, 여기서 *Metropolis* 알고리즘의 예시를 들면  
$$A(z^*, z^(\tau)) = min(1, \frac{p(z*)}{p(z^(\tau))})$$가 된다.  

수학적 수렴성 등을 제외하고, 실용적으로 왜 이러한 승인 함수가 등장했는지를 알아보기 위해 이번 포스팅의 주제 중 하나인 Posterior Sampling을 생각해보자.





Equilibrium state인 MC를 디자인해서, Required Distribution의 distribution과 일치하도록. 
어느 Chain step을 거치더라도 우리가 원하는 sample의 확률 p(z)은 변하지 않음.
만약 이게 계속 변해버리면 p와 상관 없는 샘플이 뽑혀버릴 것. 
Obtain dependent samples drawn approximately from p(z) by simulating Markov Chain
수렴성이 보장되기 때문에 그 분포의 sample뽑을 수 있다.
어느 state든 동등하게 sampling 가능
