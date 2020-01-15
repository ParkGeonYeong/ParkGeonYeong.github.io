---
title: VC generalization bound의 이해
use_math: true
classes: wide
layout: single
---   
  
Vapnik and Chervonenkis (VC) bound는 Statistical Learning Theory의 핵심이다. 
여기서는 확률론의 basic inequalities (concentration inequalities)와 hoeffding's inequality를 우선 정리하고,
hoeffding's inequality의 한계 및 infinite-function space에서의 VC bound 적용까지 살펴본다.  
  
  
**0. Concentration Inequalities**    
(구체적인 증명은 생략)  
확률론에서 어떤 empricial mean 혹은 estimation이 true mean에서 얼마나 떨어져 있는지를 알아내는 것은 굉장히 중요하다. 
Concentration Inequalities는 이와 관련된 모든 inequality를 통칭하며, 주로 1st momentum과 연관된 markov inequality, 
2nd momentum과 연관된 Chebyshev's inequality, moment generating function과 연관된 chernoff bound를 일컫으며 
chernoff bound의 special case라고 볼 수 있는 hoeffdin's inquality로 이어진다.  
- Markov Inequality
  - > If X is a nonnegative random variable and a > 0, 
  then the probability that X is at least a is at most the expectation of X divided by a:  
  $$P(X \geq a) \leq \frac{E(X)}{a}$$
  - 뒤에 나오는 chernoff bound 및 hoeffding's inequality 증명에 쓰이는 가장 기본적이자 중요한 inequality이다.
- Chebyshev's Inequality
  - > Let X (integrable) be a random variable with finite expected value μ and finite non-zero variance σ2. Then for any real number k > 0,  
  $$ Pr(\mid X - \mu \mid \geq k\sigma) \leq \frac{1}{k^2} $$
  - Chebyshev's inequality 역시 weak law of large number, VC bound 등의 증명에 활용되는 중요한 inequality이다.
- Chernoff bound
  - 위의 두 bound는 first or second-moment-based bounds으로 chernoff bound보다는 더 loose하다고 볼 수 있다. 
  - 그러나 Chernoff bound는 적률생성함수를 기반으로 exponential하게 표현되며, 이때부터 inequality가 exponential의 수혜를 입게 된다. 
  (e.g., 데이터 수 n에 exponential하게 error의 upperbound가 줄어드는) 
  - Chernoff bound의 가장 기본적인 형태는 다음과 같다. 
  - > The generic Chernoff bound for a random variable X is attained by applying Markov's inequality to $$e^{\lambda X}$$. 
  For every $\lambda >0$: 
  - $$Pr(X \geq a) \leq min_{\lambda >0}\frac{E[e^{\lambda a}]}{{e^{\lambda a}}}$$
  - 위의 minimization을 $$\lambda$$에 대해 풀어내서 optimization하는 방식을 많이 사용한다. 이를 통해 아래 딸려오는 preposition을 얻을 수 있다.
  - > Suppose $$X_1,...X_n$$ are i.i.d from Bernoulli(m). Then, 
  $$Pr(X_1 + ... + X_n \geq n(m+\delta)) \leq exp(-nKL(m+\delta \parallel m)$$
  - 위 식을 보면 슬슬 감을 잡을 수 있는데, [0, 1]로 bounded된 random variable X의 empirical mean과 true mean의 차이를 exponential bound로 표현했다. 
  - ML에서 쓰는 empirical risk minimization에서 정의하는 $$R_n(h)$$와 true risk $$R(h)$$에 비슷하게 적용할 수 있을 것이다.
**1. Hoeffding's inequality**  
Hoeffding's inequality는 learning theory에서 가장 중요한 식이라고 많이 불리는데, 
실제로 'with probability 1-d, ...'으로 시작되는 많은 ML theorem들은 대부분 hoeffding's inequality에 의해 직간접적으로 증명된다. 
- 어떤 bounded r.v. $$Z_i \in [a,b]$$가 있을 때, 
$$\mathbb{P}(\frac{1}{n} \sum_{i=1}^{n} (Z_i - \mathbb{E}[Z_i]) \geq t) \leq exp(\frac{-2nt^2}{(b-a)^2})$$ 
- 이는 $$Z_i$$의 empricial mean이 unbiased라는 가정 하에, 위의 chernoff inequality 기본 꼴에서 시작하여 최적의 $$\lambda$$를 찾아내어 증명할 수 있다.
- 단 그 과정에서 Hoeffding's lemma를 사용해야 하는데, 이는 다음과 같다.  
$$\mathbb{E}[exp(\lambda (Z-\mathbb{E}[Z]))] \leq exp(\frac{\lambda^2 (b-a)^2}{8})$$  
Hoeffding's inequality는 [a,b]를 [0,1]로 고정하고 $$Z_i$$를 binary classification의 output으로 가정하여 empiricial risk minimization에 많이 쓰인다. 
정확히 반대 방향도 그대로 증명할 수 있는데, 이를 합쳐서 절댓값 형태로 가장 많이 사용한다.   
또한 위 식에서 righthand를 $$\delta$$로 표현하면, empirical mean이 실제 true mean과 얼마 만큼 떨어져 있는지를 특정 Probability $$1-\delta$$로 표현할 수 있다. 
위의 $$E[Z]$$를 어떤 함수 f에 대한 True risk $$R(f)$$라 하고, empirical mean을 $$R_n(f)$$라고 하자.
이때 $$\epsilon$$을 $$\delta$$로 표현하면 다음 식이 성립한다:  
$$R(f) \leq R_n(f) + \sqrt \frac {log \frac {2}{\delta}}{2n}$$  
즉 어떤 f에 대해서 empirical risk와 true risk의 관계를 데이터 n에 대해 표현한 것이다. 데이터 n을 많이 확보하며, 
empirical risk가 낮아지도록 f의 hypothesis space를 넓히면 좋은 bound을 얻을 수 있을 것이다.  
**1.1. Hoeffding's inequality의 limitation**  
단, 여기서 우리는 실제로 f의 complexity를 필요 이상으로 높일 경우 true risk와 emprical risk의 격차가 굉장히 커진다는 것을 알고 있다. 
여기서 실제 f에 대한 고려가 빠졌기 때문이다. 우리가 원하는 bound는 어떤 고정된 f에 대한 것이 아니라, $$f \in \cal{F}$$에 대해서 
가장 supremum 값에 대한 것이다. 우리가 취할 수 있는 여러 hypothesis의 set에 대한 bound가 필요한 것이다.  
이때 위 f를 $$f_i \in \left\{ f_1, ... f_N \right\}$$에 대해 각각 표현하고, 이에 union bound를 적용하면 다음과 같다.  
- $$\begin{align} 
\mathbb{P}[\exists f \in \left\{ f_1, ..., f_N \right\}: R(f)-R_n(f) \geq \epsilon] 
&\leq \sum P(R(f)-R_n(f) \geq \epsilon) \\
&\leq Nexp(-2n\epsilon^2)
\end{align}$$  
  
그러나 여기서도 finite function set을 정의했기 때문에 infinite function space로 확장이 불가능하다는 치명적인 단점이 존재한다.  
  
  
**2. VC Bound**  
앞 절에서 우리는 finite number of function N을 소거하고 이를 어떠한 function의 complexity metric으로 대체해야 하는 상황이다.  
우리가 갖고 있는 것은 결국 데이터이기 때문에, 다루기 어려운 function 자체보다는 
function space에서 sampling된 어떤 function이 n개의 data를 어떻게 shatter하는지를 기반으로 이를 표현할 필요성이 있다. 
이에 growth function이라는 것을 새로 정의하면 다음과 같다:  
> The growth function is the maximum number of ways into which n points can be classified by the function class: 
$$S_{\cal{F}}(n) = sup_{(z_1,...,z_n)} \mid \cal{F_{z_1, ..., z_n}} \mid$$  
**이를 통해서 VC-dimension이 도입되지 않은, VC bound의 초기 식을 구할 수 있다.**  
> For any $$\delta > 0$$, with probability at least $$1-\delta$$, 
$$R(f) \leq R_n(f) + 2\sqrt{\frac {2log S_{\cal{F}}(2n) + log(2/\delta) }{n} }$$  
- 증명 과정은 두 단계로 나뉜다. 
- 우선 기존에 우리가 갖고 놀던 $$P(R(f) - R_n(f) \geq \epsilon)$$의 경우 알 수 없는 true risk $$R$$이 끼어 있는데, 
**이를 소거하지 않고 function의 complexity나 growth function을 도입하기 쉽지 않다.** 따라서 true risk를 다른 방식으로 대체하는데, 
이때 가상의 'ghost sample' $$Z'_1, ..., Z'_n$$을 도입한다. 여기에 symmetrization lemma를 적용하는데, 이는 다음과 같다.
  - For any t: $$\mathbb{P}[sup_{f \in \cal{F}} \mid (P-P_n)f\mid \geq t ] \leq 2\mathbb{P}[sup_{f \in \cal{F}} \mid (P'_n-P_n)f \mid \geq t/2 ]$$
  - 편의상 $$R(f)$$를 $$Pf$$로 대체하였다.
  - Lemma 증명은 triangular inequality를 우선 씌워, $$(P-P_n)f \geq t$$이며 $$(P - P'_n)f \leq t/2$$인 경우 $$(P'_n - P_n)f \geq t/2$$임을 이용한다.
  - 여기서 if-then 관계로 inequality를 세운 다음, 걸리적거리는 $$\mathbb{P}((P - P'_n)f \leq t/2)$$를 chebyshev's inequality로 날린다.
- 위의 lemma를 통해 우리의 관심사를 두 데이터셋 N, N'에 대한 classifier f의 차이로 치환하였다. 
  - 여기서 다시 union bound을 적용하여 최종 bound을 구하면 다음과 같다. 
  - ![image](https://user-images.githubusercontent.com/46081019/72446875-ae5cf980-37f7-11ea-930f-2c5be15cff6d.png)  
  - 마지막에서 두번째 식은 hoeffding's inequality, 마지막 식은 growth function을 적용한 결과이다. 
  **두 번째 식에서 error of function 에 대한 확률의 Sample space를 어떤 가상의 prediction error vector v에 대한 sample space로 바꿔버렸다.** 
  이 트릭으로 인해 우리는 해당 사건의 확률을 infinite function space가 아니라 finite 'prediction error' vector space에 대해서 구할 수 있게 되었다. 
  이는 우선 함수의 output이 binary discrete하며, data가 finite하기 때문에 가능한 것이다.
  **원래 함수는 어떤 function space(e.g., Hilbert space) 상에서 infinite dimensional vector로 종종 해석하지만, 
  여기서는 단지 2n개의 finite data에 대한 값만 알면 되는 간단한 상황이다.**
- 이러한 아름다운 증명 과정을 거쳐서 우리는 이제 function의 complexity와 number of data로 upper bound의 표현을 마쳤다.
- 그 뒤로는 $$S_{\cal{F}}(h)$$를 VC-dimension으로 bound시켜, 최종적으로 VC-dimension이 포함된 bound을 얻게 된다.  
- ![image](https://user-images.githubusercontent.com/46081019/72447883-722a9880-37f9-11ea-9669-634bd0144598.png)  








