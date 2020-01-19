---
title: Statistical Learning Theory (2) - Rademacher Complexity
use_math: true
classes: wide
layout: single
---   
  
앞선 post에서 hoeffding's inequality와 VC Bound까지의 흐름을 살펴봤다. 
여기서는 VC bound보다 좀 더 tight한 rademacher complexity를 이용한 generalization error bound을 살펴본다. 
Rademacher complexity는 VC dimension 정의와는 다르게 data distribution-dependent하며, 따라서 경우에 따라 
VC-dimension보다 조금 더 계산하기 쉬울 수 있다.  
  
**0. Rademacher Variable, correlation and Complexity**    
VC-dimension은 특정 hypothesis class의 complexity를 "shattering" 개념에서 measure했다. 
**Rademacher Complexity는 살짝 다르게, 데이터와의 "correlation" 개념에서 complexity를 measure한다.** 
'어떤 임의의 랜덤한 binary data와 classifier의 prediction이 평균적으로 얼마나 correlated될 수 있는지'가 rademacher complexity라 할 수 있다. 
  
우선 empirical risk의 정의부터 시작하면:  
$$
\begin{align}
R_n(h) &= \frac{1}{n} \sum_{i=1}^{n}(\mathbb{1}(h(x_i) \neq y_i)) \\
       &= \frac{1}{n} \sum_{i=1}^{n}(\frac{1-y_i h(x_i)}{2})
\end{align}$$  
으로 쓸 수 있다. 이는 즉 $$min_h R_n(h) \Leftrightarrow max_h \sum y_i h(x_i)$$라는 의미가 된다. 
우리가 원하는 function의 complexity measure는 어떤 임의의 dataset $$\cal{D} = \left\{ (x_i, y_i) \right\}$$에 대해서 
가늠되어야 하므로, 이를 위해 어떤 임의의 random한 dataset 혹은 labelset을 정의하면 좋을 것이다. 
이를 위해 rademacher variable을 정의한다.  
$$Pr(\sigma_i = 1) = Pr(\sigma_i = -1) = 0.5$$  
현재 hypothesis에서 취할 수 있는 maximum rademacher correlation으로 empirical Rademacher complexity를 정의하면 다음과 같다:  
$$Rad_n(\cal{F}) = \cal{\hat{R_S}}(\cal{F}) = \mathbb{E}_n[sup_{f\in \cal{F}} \frac{1}{n}\sum{\sigma_i f(x_i)} ] $$  
  
이를 통해 현재 function hypothesis가 얼마나 complex한지 확인할 수 있다. 단적으로 만약 data가 n개 주어졌고, 
discrete function space $$\cal{F}$$ with $$\mid \cal{F} \mid = 2^n$$이라고 하자. 이때 주어진 data가 어떤 식이던, 
현재 function space에서 data와 완벽하게 fitting되는 function f를 찾아낼 수 있을 것이다. 따라서 rademacher complexity의 maximum value는 1이다. 
  
    
**1. Convergence Bound**  
앞서 VC-dimension 및 growth function을 활용해 VC-bound을 표현했다. 여기서는 위에서 정의한 rademacher complexity를 통해 generalization bound을 정의해 본다.   
> Thm. with probability at least $$1-\delta$$, $$R(f) \leq \hat{R}_n(f) + 2Rad_n(F) + \sqrt{\frac{log 1/\delta}{2n} }$$   

- 증명은 두 단계로, 우선 hoeffding's inequality를 적용한 다음, rademacher complexity를 적절히 도입한다.
- $$\Phi(n) = sup_f{[R(f) - R_n(f)]}$$라 하자. 
  - $$\Phi$$는 bounded이므로, Hoeffding's inequality에 의해, 
  With probability at $$1-\delta$$, $$\Phi(n) \leq \mathbb{E}_n(\Phi(n)) + \sqrt{\frac{log 1/\delta}{2n} }$$  
- 이때, 저번 VC-Bound 유도 과정과 비슷하게 true generlization error $$R$$를 없애 주기 위해 ghost sample $$n'$$을 도입한다:  
$$\begin{align}
\mathbb{E}_n[\Phi] &= \mathbb{E}_n[sup_f(R(f) - R_n(f))] \\
&= \mathbb{E}_n[sup_f( \mathbb{E_{n'}} [R_{n'}(f) - R_n(f)])] \\
&\leq \mathbb{E}_n[\mathbb{E_{n'}} [sup(R_{n'}(f) - R_n(f))]] \\ 
&\doteq \mathbb{E}_n[\mathbb{E_{n'}} [sup \sum_i \frac{1}{n} \sigma_i(R_{n'}(f_i) - R_{n}(f_i))]] \\
&\leq \mathbb{E}_n[\mathbb{E_{n'}} [sup \sum_i \frac{1}{n} \sigma_i(R_{n'}(f_i)]] + \mathbb{E}_n[\mathbb{E_{n'}} [sup \sum_i \frac{1}{n} \sigma_i(R_{n}(f_i)]] \\
&= 2Rad_n(F) $$

\end{align}
$$  
- 4번째 줄에서 rademacher variable의 도입을 위한 trick이 사용되었는데, $$R_{n'}(f) - R_{n}(f)$$에서 i번째 n, n' dataset sample을 50%의
랜덤한 i.i.d 확률로 바꿔준다고 가정한다. 이 경우 위에서 정의한 rademacher variable $$\sigma_i$$를 통해 위 식처럼 쓸 수 있다.  
- $$\Phi(n) \leq \mathbb{E}_n(\Phi(n)) + \sqrt{\frac{log 1/\delta}{2n} }$$의 $$\mathbb{E}_n(\Phi(n))$$를 substitute하고, $$\Phi$$에  
$$R(f) - R_n(f)$$를 적용해 주면 우리가 원하는 최종 bound을 얻을 수 있다.  

  
**2. Summary**   
이전 포스트에서 구한 VC-bound에 이어, 여기서는 distribution-dependent measure인 rademacher complexity를 도입해 
generalization error bound을 구해 보았다. 두 bound 모두 결국은 n이 무한대에 이름에 따라 data error term은 떨어져 나가고, 
estimation error term만 남는 것을 알 수 있다. 이 때 estimation error term은 hypotheseis space의 complexity를 무작정 높인다고 해서 
낮출 수 없는데, 이는 empirical error는 줄어들더라도 VC-dimension 혹은 rademacher complexity는 늘어나기 때문이다. 
따라서 적당한 trade-off를 찾아내어 well-generalizable function space를 설정하는 것이 중요하다.  
시간이 남을때 두 bound을 활용하고 있는 논문 예시들을 정리해 보려 한다.

