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

**1. Sampling Methods**   
두 방식 모두 given $$z$$에 대해서 $$p(z)$$를 계산할 수 있다고 가정한다.  

**1.1 Rejection Sampling**  
Rejection Sampling은 이름에서 알 수 있듯이 Sample을 제안해 특정 기준을 obey하지 못하는 경우 reject하여 underlying true distribution을 
근사한다. 우선 우리의 관심사인 분포를 $$p(z)$$이라 하고, sample의 proposal distribution $$q(z)$$을 정의한다. 
이때 $$q(z)$$에 어떤 k를 곱해, $$p(z)$$를 envelope하는 $$kq(z)$$를 얻는다고 하자. 그림으로는 다음과 같다.  
![images](https://user-images.githubusercontent.com/46081019/55325263-73374700-54bf-11e9-98bc-63ee69fbd6c3.png)  
즉 q(z)를 following하는 $$z_0$$를 샘플링한 다음, $$\frac{p(z)}{kq(z)}$$의 확률을 통해 reject 여부를 결정한다. 
따라서 회색 영역에 포함되는 sample은 reject되고, 흰색 영역만이 남는다. 해당 면적의 확률을 계산하면 다음과 같다.  
$$\int[{p(z)/kq(z)} ]q(z)dz = \frac{1}{k}\int{p(z)}dz$$  
즉 k를 크게 할 경우 reject ratio는 높아지게 되며, 이를 최소화하는 이상적인 분포를 찾아내는 것의 어려움이 있다. 
**이는 특히 분포의 dimensionality가 높아짐에 따라 큰 문제가 된다.** 
높은 dimensionality에서 k가 조금만 높아져도 rejection ratio는 크게 높아지기 때문에, 효율성의 문제가 생긴다.  

예를 들어 $$p(x)$$가 zero-mean D-dimensional 가우시안 분포인 경우를 생각하자. 굉장히 이상적으로 $$q(x)$$를 동일한 zero-mean 가우시안으로 설정하면, 
$$\sigma_q^2 > \sigma_p^2$$ 조건을 만족해야만 $$q(x)$$가 rejection sampling에서 사용될 수 있다.  
이때 k는 $$k=({\sigma_q}/{\sigma_p})^D$$이 된다고 한다. 즉 매우 파라미터가 고차원인 분포의 경우, 
k는 기하급수적으로 올라가며 따라서 승인률은 매우 낮아질 수 밖에 없다.  

**1.2 Importance Sampling**  
Importance Sampling은 rejection sampling처럼 $$q(z)$$를 사용하는 점은 비슷하지만, 
sample data에 직접적으로 관여한다기보다 이를 통해 얻을 수 있는 metric(expectation 등)에 집중한다. 
가령 dimension이 높거나 probability mass가 불균형할 경우 uniform sampling만으로 정확한 대표값을 얻기 어렵다. 
따라서 proposal distribution $$q(z)$$를 이용해 비간접적으로 이를 얻는데, 과정은 다음과 같다.  
$$E[f] = \int{f(z)p(z)}dz \\ 
=\int{f(z)\frac{p(z)}{q(z)}}q(z)dz \\
\cong\frac{1}{N}\sum_{n=1}^{n=N}\frac{p(z^n)}{q(z^n)}f(z^n)$$   

이때 $$p(z)$$와 $$q(z)$$의 ratio을 importance weights라고 정의한다. 
Rejection Sampling과 마찬가지로 Importance Sampling 역시 $$q(z)$$의 정확한 선택이 중요하다. 
Complex하고 불균형한 $$p(z)$$와 $$f(z)$$가 있을 때, $$q(z)$$가 만약 전혀 특정 영역에 대한 sample을 뽑지 못한다면 
importance weights의 variability는 낮지만 실제 expectation과는 매우 거리가 있는 값을 추정할 수 있다. 
따라서 최소한 $$p(z)$$가 significatn한 구간에서 $$q(z)$$값이 매우 작거나 0이 되면 안 된다는 조건이 발생한다.  
  
비슷한 문제가 강화 학습의 off-policy value 추정에서도 발생할 수 있다. (보다 자세한 내용은 Richard Sutton 책의 Chap 5.5)  
Target과 behavior policy를 정의하여 Monte-Carlo 추정을 실행할 경우 MC trajectory가 길어질 수록 importance weights는 굉장히 여러 번 곱해지기 때문에 높은 variability를 갖게 된다. 또한 MC가 아닌 TD 버전의 경우에도 target policy와 behavior policy가 one-step에 대해서만 비슷해져도 된다는 문제점이 발생한다. 따라서 이러한 importance sampling의 단점을 피하고자 Bellman Optimal Equation의 unbiased sampling을 통해 문제를 해결하는 알고리즘이 Q-learning이다. 

**2. Markov Chain Monte Carlo**  

MCMC는 앞선 Sampling 알고리즘이 고차원 분포에서 좋지 못한 성능을 보이는 것을 개선할 수 있는 알고리즘이다.  

**MCMC = Obtain dependent samples drawn approximately from p(z) by simulating Markov Chain**  

우선 Markov Chain에 대해서 간단히 짚고 넘어가자.  

**2.1 Markov Chain**  
Markov Chain이란 current state에 대한 non-deterministic 확률 분포를 얻는 과정으로, 
매 시간마다 state-space는 바뀌거나 혹은 같은 상태를 유지할 수 있다. 
이때 만약 시간 흐름에 관계 없이 stationary한 평형 상태를 계속해서 유지한다면 이를 stationary state라고 부른다. 
즉 어떤 특정 state에 대한 확률을 $$\pi(z)$$라고 간단히 하고, 마르코프 체인의 전환 함수 $$T(z'{\leftarrow}z)$$이라 할 때
$$\pi(z) = \pi(z)T(z'{\leftarrow}z)$$가 만족된다면 이를 stationary하다고 부른다.  
**이때 Markov chain이 irreducible하고 ergodic한 경우 $$\pi(z)$$은 무한한 transition n에 대해서 stationary하게 수렴하게 된다.** 
- Irreducible : Possible to get to any state from any state. $$T$$의 조건부 확률이 0이 되면 안된다.
- Ergodic : aperiodic 성질과 recurrent 성질을 합침. 
  - Aperiodic : Any return to state가 어떤 k step의 배수로 일어나는 경우를 periodic하다고 하며, 그 반대를 aperiodic으로 정의
  - Recurrent : 모든 state가 transient하지 않은 경우. Return to state가 보장됨  
또한 stationarity의 충분 조건으로 markov chain의 reversiblity가 있다. 
즉 $$\pi(z)T(z'{\leftarrow}z) = \pi(z')T(z{\leftarrow}z')$$인 경우 반드시 state distribution은 stationary하다. 
어떤 state에서 다른 state로 갈 때 그 분포가 유지되며 모두 방문 가능하기 때문에 직관적으로 받아들일 수 있다.

**2.2 MCMC의 직관적 이해**  
MCMC은 마르코프 체인의 수렴성을 수학적 백그라운드로 갖고 있는 알고리즘이다. Equilibrium distribution이 우리의 관심사 분포와 일치하는, 
가역적인 마르코프 체인을 설계해서 그 안에서 계속해서 안정적으로 샘플링을 진행하는 것이다. 
조금 더 쉽게 말하면 $$p(z)$$가 stationary하도록 어떤 마르코프 체인의 전환 $$q(z'{\leftarrow}z)$$을 설계하면, 이를 기반으로 계속 시뮬레이션을 돌려 모든 state에 대한 샘플을 보장 받을 수 있다. 같은 $$q$$에서 눈치 챘을 수 있지만 MCMC 역시 별도의 $$q(z)$$를 정의하며, 
이 때 앞선 알고리즘과의 차이점은 현재 state $$z(\tau)$$를 기준으로 $$q(z)$$가 다음 $$z(\tau')$$ state를 결정한다.
  
조금 더 구체적으로 살펴보자. 설계한 마르코프 체인의 Equilibrium distribution이 우리의 타겟 분포와 일치하는 경우, 어떤 chain step을 거쳐 샘플링을 하더라도 우리의 타겟인 분포 $$p(z)$$은 stationarity를 유지할 수 있다.  
$$p(z(\tau)) = p(z(\tau-1)) = ... \equiv p_eq(z)$$  
만약 해당 조건이 성립하지 않을 경우 특정 chain step에 따라 $$p(z)$$는 non-stationary해질 수 있고 따라서 올바른 샘플링이 어렵다. 
반대로 성립할 경우에는 마르코프 체인에서 어떤 state로 갈 확률이 항상 동일하게 유지되기 때문에 균일한 방식의 샘플링을 유지할 수 있다.
이러한 stationarity가 만족되도록 위에서 전환이자 승인 함수 $$Q$$를 설계했기 때문에 모든 $$p$$의 state를 방문할 수 있는 것이 보장되고, 
계속된 샘플링 시뮬레이션 끝에 invariant distribution $$p$$로 수렴할 수 있다.
  
**그럼 이 과정이 만족되려면 2.1에서 어떤 수식이 만족되어야 할까? 위의 전환 함수를 $$T(z'{\leftarrow}z)$$이라 생각할 때$$p(z)T(z'{\leftarrow}z) = p(z')T(z{\leftarrow}z')$$가 성립하면 된다. (가역적인 markov chain -> Stationarity 만족)**  
이제 이를 위해서, 제안된 새로운 state $$z'$$를 승인할지 결정하는 함수 $$A(z', z(\tau)$$를 생각하자. 
만약 $$p(z')T(z{\leftarrow}z')$$가 $$p(z)T(z'{\leftarrow}z)$$보다 크다면, $$p(z)T(z'{\leftarrow}z)$$의 값을 증가시키는 과정이 필요하다.따라서 이번 trial에서 무조건 $$z'$$을 승인한다. 반대라면 확률비례적으로 $$p(z)T(z'{\leftarrow}z)$$을 줄인다. 따라서 $$A$$는 다음과 같다.  
$$A(z^*, z^{\tau}) = min(1, \frac{p(z')T(z{\leftarrow}z')}{p(z)T(z'{\leftarrow}z)})$$  
  
알고리즘적으로 보자. 임의의 초기 state $$z(0)$$을 뽑고, 해당 state에서 다시 분포 $$T$$를 이용해 새로운 state $$z'$$를 샘플링했다고 하자. 
위 승인 함수에 따라 transition 여부를 결정한다. 이를 무한히 반복한다.  
   
수학적 수렴성 등을 우선 제외하고, 실용적으로 왜 이러한 승인 함수 개념이 등장했는지를 알아보기 위해 이번 포스팅의 주제 중 하나인 Posterior Sampling을 생각해보자.
$$P(\thetaㅣx) = \frac{P(xㅣ\theta)P(\theta)}{P(x)}$$에서 우리의 관심사인 $$P(\thetaㅣx)$$를 $$\frac{P(xㅣ\theta)P(\theta)}{P(x)}$$으로 생각하자. State $$\tau$$에서의 $$\theta(\tau)$$를 생각하자. 그리고 현재 모인 데이터에 대해서 마르코프 체인의 transition 함수를 $$P(xㅣ\theta)$$이라 생각하자.  

이때 새로운 state에서의 후보 sample인 $$\theta^*$$를 뽑자. 이를 활용해서 동일한 방식으로 사후 분포를 샘플링하면 
$$\frac{P(xㅣ\theta^*)P(\theta^*)}{P(x)}$$으로 표현된다. 
Markov chain에서의 두 state를 비교해서 어떤 결과가 사후 분포를 더 잘 설명하고 있는지 보려면, 두 sampled value의 ratio을 비교한다.
$$A(z^*, z^{\tau}) = min(1, \frac{(* 결과)}{(\tau 결과)})$$  
이때 가능한 결과는 총 2가지인데, 만약 $$1<\frac{(* 결과)}{(\tau 결과)}$$이라면 이 후보 샘플을 새로운 state로 확정짓고 다시 과정을 반복한다. 
만약 반대라면, [0, 1] uniform distribution의 랜덤 샘플값 u와 위 결과를 비교하여 state를 확정지을지 여부를 결정한다.
이는 설령 새로운 state가 사후 분포의 확률을 높이지 못했더라도, 어쨌든 전체 분포를 샘플링하여 estimate해야하기 때문에 exploration 과정은 필요하기 때문이다. 이 과정을 통해 state는 다양한 범위에 걸쳐 바뀌게 되고 실제 사후 분포를 더 잘 설명할 수 있게 된다.  

![image](https://user-images.githubusercontent.com/46081019/54412028-98dcf600-4734-11e9-9798-576f8893dede.png)  
위 그림은 multivariate gaussian 분포 (타원형)을 제안 분포삼아 sampling하는 과정이다. 
초록색 승인된 샘플을 보면 관심 분포 상에서 확률이 높은 state로 이동하는 경우가 많지만 반드시 모든 경우에서 그렇지는 않다.  
  
같은 얘기를 반복하고 있지만, q(z'|z)의 제대로 된 가정은 굉장히 중요하다. 가령 q의 variance를 굉장히 크게 잡았다고 가정하면, 새로운 state $$z'$$을 
굉장히 폭 넓게 sampling하여 true underlying distribution $$p(z)$$의 구석구석을 샘플링할 수 있다. 그러나 그만큼 데이터 샘플링의 resolution 혹은 density는 낮아지게 된다. 반대로 너무 variability를 낮추면 accept되는 sample수는 더 많을지 몰라도 너무 특정 지역에 몰리거나 slow random walk의 행동 패턴을 보이게 될 것이다.
