---
Title : "Markov Chain Monte Carlo와 Posterior Sampling"
use_math : true
layout : single
classes : wide
---

**0. Posterior Sampling의 필요성**  
Bayes Inference에서 posterior distribution을 정확히 구하는 것은 계산 측면에서 매우 어려운 일이다. 
$$P(\thetaㅣx) = \frac{P(xㅣ\theta)P(\theta)}{P(x)}$$에서 분모의 evidence $$P(x)$$를 marginalization 과정으로 계산해야하기 때문이다.
모든 가능한 $$\theta$$에 대해 이를 계산하는 것은 practical하게는 불가능한 일이다. Analytically intractable한 경우가 많다. 

따라서 우리가 원하는 posterior distribution을 sampling 과정으로 근사해서 구하는 과정(approximate inference)을 생각해 볼 수 있는데, 
많이 쓰이는 알고리즘이 바로 Markov Chain Monte Carlo(MCMC)이다. 여기서는 우선 몇몇 sampling 기법을 먼저 다룬 다음, 
해당 과정들의 문제점을 보완할 수 있는 MCMC를 설명하겠다.  

**1. 다양한 Sampling Methods**  

**1.1 Standard Sampling**  

**1.2 Rejection Sampling**  
**1.3 Importance Sampling**  
**2. Markov Chain Monte Carlo**  

MCMC는 앞선 Sampling 알고리즘이 고차원 분포에서 좋지 못한 성능을 보이는 것을 개선할 수 있는 알고리즘이다.  
**MCMC = Obtain dependent samples drawn approximately from p(z) by simulating Markov Chain**
  
가령 rejection sampling에서 제안 분포 $$q(z)$$을 먼저 결정할 때, 우리 관심사인 $$p(z)$$와 매우 유사하도록 만들어야 승인률이 올라간다. 
예를 들어 $$p(x)$$가 zero-mean D-dimensional 가우시안 분포인 경우를 생각하자. 굉장히 이상적으로 $$q(x)$$를 동일한 zero-mean 가우시안으로 설정하면, 
$$\sigma_q^2 > \sigma_p^2$$ 조건을 만족해야만 $$q(x)$$가 rejection sampling에서 사용될 수 있다.  
이때 k는 $$k=({\sigma_q}/{\sigma_p})^D$$이 된다고 한다. 즉 매우 파라미터가 고차원인 분포의 경우, 
k는 기하급수적으로 올라가며 따라서 승인률은 매우 낮아질 수 밖에 없다. 결국 샘플링의 효율은 너무 떨어지며, 이는 Importance Sampling에서도 비슷하다. 
  
이를 해결하기 위한 방식이 MCMC이다. Markovian 특성, Monte Carlo 등은 따로 설명하지 않고 알고리즘의 특성과 증명에 집중하겠다. 
굉장히 복잡하거나 사전 지식이 없지만, 주어진 $$z$$에 대해 평가하는 것은 가능한 $$p(z)$$이 있다고 하자. 이를 대상으로 앞서 rejection sampling과 importance sampling 때처럼 제안 분포 $$q(z)$$을 설정한다. 이 $$q$$를 이용하여 $$z^{\tau}$$를 sampling하게 되는데, 
이를 markov chain에서의 state at time $$\tau$$로 사용하는게 알고리즘의 시작이다. 
임의의 초기 state $$z(0)$$을 뽑고, 
해당 state에서 다시 분포 q를 이용해 새로운 state $$z^*$$를 샘플링했다고 하자. 
이때 MCMC에는 rejection sampling과 유사하게, 이 새로운 state의 인정 여부를 결정하는 승인 과정을 거친다. 
알고리즘마다 다르지만, 여기서 *Metropolis* 알고리즘의 예시를 들면  
$$A(z^*, z^{\tau}) = min(1, \frac{p(z*)}{p(z^(\tau))})$$가 된다.  
  
수학적 수렴성 등을 우선 제외하고, 실용적으로 왜 이러한 승인 함수가 등장했는지를 알아보기 위해 이번 포스팅의 주제 중 하나인 Posterior Sampling을 생각해보자.
$$P(\thetaㅣx) = \frac{P(xㅣ\theta)P(\theta)}{P(x)}$$에서 우리의 관심사인 사후분포를 $$P(\thetaㅣx)$$로 두고, 
(혹은 $$\frac{P(xㅣ\theta)P(\theta)}{P(x)}$$와 동일), state $$\tau$$에서의 $$\theta(\tau)$$를 생각하자.  
이때 현재 모인 데이터 x에 대해서 사전분포를 통해 다시 샘플링된 사후 분포는 $$\frac{P(xㅣ\theta(\tau))P(\theta(\tau))}{P(x)}$$으로 표현된다. 
이때 새로운 state에서의 후보 sample인 $$\theta^*$$를 생각하자. 이를 활용해서 동일한 방식으로 사후 분포를 샘플링하면 
$$\frac{P(xㅣ\theta^*)P(\theta^*)}{P(x)}$$으로 표현된다. 
Markov chain에서의 두 state를 비교해서 어떤 결과가 사후 분포를 더 잘 설명하고 있는지 보려면, 두 sampled value의 ratio을 비교한다.
$$A(z^*, z^{\tau}) = min(1, \frac{(* 결과)}{(\tau 결과)})$$  
이때 가능한 결과는 총 2가지인데, 만약 $$1<\frac{(* 결과)}{(\tau 결과)}$$이라면 이 후보 샘플을 새로운 state로 확정짓고 다시 과정을 반복한다. 
만약 반대라면, [0, 1] uniform distribution의 랜덤 샘플값 u와 위 결과를 비교하여 state를 확정지을지 여부를 결정한다.
이는 설령 새로운 state가 사후 분포의 확률을 높이지 못했더라도, 어쨌든 전체 분포를 샘플링하여 estimate해야하기 때문에 exploration 과정은 필요하기 때문이다. 이 과정을 통해 state는 다양한 범위에 걸쳐 바뀌게 되고 실제 사후 분포를 더 잘 설명할 수 있게 된다.  

![image](https://user-images.githubusercontent.com/46081019/54412028-98dcf600-4734-11e9-9798-576f8893dede.png)  
위 그림은 multivariate gaussian 분포 (타원형)을 isotropic gaussian distribution을 제안 분포삼아 sampling하는 과정이다. 
초록색 승인된 샘플을 보면 관심 분포 상에서 확률이 높은 state로 이동하는 경우가 많지만 반드시 모든 경우에서 그렇지는 않다.  
   
MCMC은 마르코프 체인의 수렴성을 수학적 백그라운드로 갖고 있는 알고리즘이다. Equilibrium distribution이 우리의 관심사 분포와 일치하는, 
가역적인 마르코프 체인을 설계해서 그 안에서 계속해서 안정적으로 샘플링을 진행하는 것이다. 
조금 더 쉽게 말하면 $$p(z)$$가 stationary하도록 마르코프 체인의 전환 $$T(z'\leftarrowz)$$을 설계한 뒤, 이를 기반으로 계속 시뮬레이션을 돌려 
샘플을 얻는다.  
  
우선 결론을 먼저 보면, 설계한 마르코프 체인의 Equilibrium distribution이 우리의 타겟 분포와 일치하는 경우, 어떤 chain step을 거쳐 샘플링을 하더라도 우리의 타겟인 분포 $$p(z)$$은 stationarity를 유지할 수 있다.
$$p(z(\tau)) = p(z(\tau-1)) = ... \equiv p_eq(z)$$
해당 조건이 성립하지 않을 경우 특정 chain step에 따라 $$p(z)$$는 non-stationary해질 수 있고 따라서 올바른 샘플링이 어렵다. 
반대로 성립할 경우 마르코프 체인에서 어떤 state로 갈 확률이 항상 동일하게 유지되기 때문에 균일한 방식의 샘플링을 유지할 수 있다.
이러한 stationarity가 만족되도록 위에서 전환이자 승인 함수 $$A$$를 설계했기 때문에 모든 $$p$$의 state를 방문할 수 있는 것이 보장되고, 
계속된 샘플링 시뮬레이션 끝에 invariant distribution $$p$$로 수렴할 수 있다.
