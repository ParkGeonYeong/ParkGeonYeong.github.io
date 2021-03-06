---
title : "k-means, EM, Variational-GMM, Dirichlet Process 정리"
use_math : true
layout : single
classes : wide
---

Clustering은 Machine Learning에서도 unsupervised learning의 대표적인 갈래이다. 
Unlabeled data를 군집화하고, 각 군집의 특성을 파악하는 과정이다. 
이러한 군집은 단일 분포로는 표현이 어렵기 때문에 Mixture Model을 사용하게 된다.  
  
  
Clustering의 가장 간단한 알고리즘은 K-means Clustering이다. 
kNN의 k-nearest neighbor와 K-means Clustering의 k는 다르다. 후자는 전체 클러스터의 개수를 의미하는 하이퍼 파라미터이다.  
K-means Clustering은 cluster assign을 'hard'하게, 혹은 'deterministic'하게 한다.  
많은 머신러닝 알고리즘이 그렇듯 이를 점점 확률적으로 개선해 나가는데, 
cluster를 'soft'하게 assign하는 알고리즘이 EM을 이용한 GMM 방식이다. 

  
한편 Cluster의 개수 k를 미리 정해주어야 한다는 점 역시 K-means clustering의 단점이고 이는 EM을 통해서 개선할 수 있는 범위 밖의 문제이다. 
이를 해결하기 위해 Variational Inference를 이용한 GMM, 그리고 non-parameteric 방식의 dirichlet process 알고리즘을 사용한다.   
참고로 Variational Inference를 활용한 방식은 간접적인 방식이다. 
여기서는 순서대로 K-means, Gaussian Mixture, EM을 다룬 다음 여력이 된다면 Variational Inference(mean-field), dirichlet process까지 알아보고자 한다.    
  
  
**0. K-means Clustering**  
![image](https://user-images.githubusercontent.com/46081019/54875301-ba7a6380-4e3f-11e9-9fe1-066642621807.png)  
- K-means Clustering은 K개의 centroid $$\mu_k$$을 parameter로 정의하고, 다음 loss에 대해 업데이트한다.  
  - $$L = \sum_n\sum_k \gamma_{nk} \parallel x_n-\mu_k \parallel^2$$
  - 이때 $$\gamma_{nk}$$은 n-th data point에 대한 cluster k의 responsibility이다. 
  - K-means Clustering은 hard assign이기 때문에 $$\gamma$$가 binary이다. 
  - EM에서는 이를 0과 1 사이의 확률값으로 사용한다.
- 하이퍼 파라미터 $$\gamma$$와 $$\mu_k$$가 서로 entangled 관계이기 때문에 iterative하게 풀게 된다. 
  - 우선 \mu를 임의로 초기화하고, 이를 기반으로 $$\gamma$$를 설정한 다음, 다시 $$\mu_k$$에 대해 loss를 최적화
  - 이때 $$\mu_k$$에 대해 loss를 partial derivation하면, $$\mu_k = \frac{\sum_n \gamma_{nk}x_n}{\sum_n \gamma_{nk}}$$
  - 현재 $$\gamma$$는 binary이므로, 이는 k-th cluster에 포함된 모든 데이터의 평균이다.  
    
    
    
**1. Gaussian Mixtures and EM**  
**1.1. Gausian Mixtures**  
- Pdf of GMM
  - $$p(x) = \sum_k \pi_k N(x \mid \mu_k, \Sigma)$$
- 여기서 \pi는 mixture parameter로써, x에 대한 k-th Gaussian 분포의 responsibility이다.
  - $$\sum_k \pi_k = 1$$
- 이때 parameter 외에, latent variable z을 추가로 정의하자.
  - $$p(z_k = 1) = \pi_k$$
  - z는 one-hot-like variable이므로, 다음과 같이 쓸 수 있다.  
  - $$p(z) = \prod_k{\pi_{k}^{z_{k}}}$$
  - Latent Variable을 도입하는 것이 그냥 $$\pi_k$$를 쓰는 것보다 더 복잡해 보이지만, 이후 EM을 전개할 때 수식적으로 많은 이득을 본다.
- 어떤 k-th distribution에 속한 x에 대해 probability density는 다음과 같다.
  - $$p(x \mid z_k = 1) = N(x \mid \mu_k, \Sigma_k)$$
  - 혹은 전체 z에 대해서 $$p(x \mid z) = \prod{N(x \mid \mu_k, \Sigma_k)^{z_k}}$$
- 데이터 x에 대한 marginalized likelihood는 다음과 같다. 
  - $$p(x) = \sum p(z)p(x \mid z) = \sum \pi_k N(x \mid \mu_k, \Sigma_k)$$
- 또한 앞서 k-means에서처럼 conditional probability of z given x, 혹은 responsibility $$\gamma(z_k)$$를 정의할 수 있다. 
  - $$\begin{equation} 
\begin{split}
\gamma(z_k) &= p(z_k = 1 \mid x) \\
&= \sum \frac{p(z_k=1)p(x \mid z_k = 1)}{\sum_j p(z_j=1)p(x \mid z_j = 1)} \\
&=\frac{\pi_kN(x \mid \mu_k, \Sigma_k)}{\sum_j \pi_jN(x \mid \mu_j, \Sigma_j)}
\end{split}
\end{equation}$$
  
- 개별 데이터가 아닌 전체 $$X$$에 대해서 log likelihood는 다음과 같다. 
  - $$ln p(x) = \sum_{i=1}^N lnp(X \mid \pi, \mu, \Sigma) = \sum_{i=1}^N ln \sum_k \pi_k N(x_i \mid \mu_k, \Sigma_k)$$
  - Logarithm 안에 summation이 있어 계산이 어렵다는 문제점이 있다. 
  - 이는 logarithm의 concave 특성을 이용해 EM에서 해결하게 된다. 
  - 그 전에 앞으로 이 GMM의 likelihood을 optimize하는 과정에서 나타날 수 있는 singularity 문제에 대해 간단히 짚고 넘어간다.
- ![Figure9 7](https://user-images.githubusercontent.com/46081019/62991124-55c0b380-be89-11e9-9d6c-f1c078b48fcc.png)  
  - 위 figure에서 보이다시피 어떤 j-th cluster의 평균 $$\mu_j$$와 특정 data point $$x_n$$이 일치하는 경우를 생각하면, $$x_n$$에 대한 likelihood는 다음과 같다. 
  - $$N(x_n \mid x_n, \sigma_j^2I) \propto \frac{1}{\sigma_j}$$
  - 따라서 figure처럼 굉장히 narrow한 data에 대해 표현되는 distribution이 mixture에 끼어 있다면, $$\sigma_j$$가 작음에 따라 전체 likelihood는 발산해버릴 수 있다.  
  - 이를 특이점 문제라 부르며, gradient-based optimization 과정에서 이를 피하기 위한 테크닉이 필요하다.
  - 가령 특정 gaussian 분포가 collapse하는 임계점에 도달하면 다시 평균과 공분산을 initialize하여 새로운 local minima에 빠지도록 유도한다.

**1.2.1 Concept of EM and Evidence Lower Bound**      
- Expectation-Maximization 알고리즘(EM)은 parameter와 latent variable의 교차 최적화를 통해, Maximum likelihood solution을 찾아낸다.
  - 이때 Expectation step은 old-parameter에 대해 latent variable을 re-compute, Maximization step은 given latent variable에 대해 new-parameter를 estimate
  - local minima에 빠질 가능성이 높고 iteration도 많이 돌아야 하지만, 수렴성이 증명되어 유용함. 
- Evidence Lower Bound의 conceptual derivation
  - 앞서 likelihood가 logarithm of summation 형태임을 해결하기 위해, logarithm funciton이 concave임을 활용하여 jensen's inequality 적용.  
  - $$\begin{equation}
\begin{split}
likelihood  
  &= ln(\sum_{z} P(x,Z\mid \theta)) \\ 
  &= ln(\sum_{z}q(Z)\frac{P(x,Z \mid \theta)}{q(Z)})
  \end{split}
 \end{equation}$$  
  - $$P(x, z)$$에 대해 쓴 꼴은 likelihood의 complete 꼴이라 부름 (1.2.2 참고)
  - 참고로, 다른 방식으로도 적용 가능
    - $$ln(\sum_{z}q(Z)\frac{P(x,Z \mid \theta)}{q(Z)})\geq\sum_{z}q(Z)ln(\frac{P(x,Z \mid \theta)}{q(Z)}) $$
    - 우변의 분모, 분자를 분리하면 $$\sum_{z}q(Z)ln(P(x,Z \mid \theta))-\sum_{z}q(Z)ln(q(Z)) = E_{q(Z)}ln(P(x,Z \mid \theta))+H(q)$$
- 첫 번째 식에서 log-likelihood의 lower-bound를 tight시킬 조건은 latent variable의 distribution $$q(Z)$$가 $$P(Z \mid x, \theta)$$와 일치할 때
  - 따라서 완벽히 학습되지 않은 old-parameter $$\theta$$를 기반으로 우선 $$q(Z)$$를 구하고, 
    - (첫 번째 Lower bound tighten)
  - 다시 q(Z)를 통해 parameter $$\theta$$를 $$argmax_{\theta}E_{q^0(Z)}ln(P(x,Z \mid \theta^0)$$으로 업데이트
    - (두 번째 Lower bound tighten)
- 이러한 두 변수 $$q(Z), \theta$$의 interaction을 통해 log-likelihood의 Lower Bound을 꾸준히 maximize할 수 있음
  
  
**1.2.2 EM for GMM**  
- $$ln p(x) = \sum_{i=1}^N lnp(X \mid \pi, \mu, \Sigma) = \sum_{i=1}^N ln \sum_k \pi_k N(x_i \mid \mu_k, \Sigma_k)$$
- $$\pi_k$$는 일단 무시하고, 모델 파라미터 $$\mu_k, \Sigma_k, \pi_k$$에 대해서 partial differentiation을 취해보면 다음과 같다.
  - $$\mu_k, \Sigma_k$$의 경우 단순히 multivariate gaussian의 differentiation
  - $$\mu_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik})x_i$$
  - $$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(z_{ik})(x_i - \mu_k)(x_i - \mu_k)^T$$
  - 결과를 보면 k-th cluster에 각 i-th data point가 갖는 responsibility를 weight로 하여, sample mean과 sample covariance를 구한 것
    - 굉장히 직관적이다.
  - $$\pi_k$$의 경우 $$\sum \pi_k = 1$$ 라그랑지안 조건을 걸어준다. 
  - 미분을 취한 다음 $$\pi_j$$을 곱해 j에 대해서 summation하는 트릭을 사용해 라그랑지안 계수를 구한다. 
  - 결과적으로 $$\pi_k = \frac{N_k}{N}$$이 된다. 
- 이때 모든 term에 대해서 $$\gamma$$가 포함되어 있기 때문에 closed-form solution을 구할 수 없다.
- 따라서 앞서 언급하였듯이 iterative algorithm을 사용한다. 
- 이 과정은 결국 latent variable $$z$$를 old-parameter $$\theta_{old}$$에 대해서 추정하고, 이 정보와 observation을 결합하여 새로운 $$\theta^{new} = argmax_{\theta} \sum_z p(z \mid x, \theta^{old})p(x, z \mid \theta)$$을 찾아내는 것이라 할 수 있다. 
- $$p(x, z)$$을 complete-data log likelihood라고 한다.
  - 결국 latent z을 꾸준히 inference하여, 각 z에 대한 평균적인 complete-data log likelihood $$p(x, z)$$를 최대화하려는 노력이다.  
  
- 이렇게 latent variable $$z$$를 inference하는 관점에서 위의 GMM 과정을 다시 살펴보자.
- $$ln p(x, z \mid \mu, \Sigma, \pi) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} [{ln \pi_k + lnN(x_n \mid \mu_k, \Sigma_k)}]$$
- $$\pi$$ 외에 $$z$$ latent variable을 추가해서 식을 정리하였더니 logarithm 꼴이 보다 깔끔하게 바뀐 것을 알 수 있다.
  - 이것이 latent variable $$z$$를 도입하여 얻을 수 있는 효과이다. 
  - 이때 똑같이 $$\pi_k$$에 대해서 라그랑지안 제약을 걸고 식을 전개하면 아래 결과를 얻을 수 있다.
  - $$\pi_k = \frac{1}{N} \sum_{n=1}^N z_{nk}$$
  - 즉 mixing coefficient $$\pi$$가 실제로 얼마나 많은 데이터가 k-th cluster에 assign 되었는가에 대한 비율이라 할 수 있다.
- 이때 모델이 현재 추정한 $$z_{nk}$$들에 대해서 expectation을 취하면 $$\gamma(z_{nk})$$을 얻을 수 있다. 
- 이에 기반하여 추정한 z에 대해 평균적인 complete-data log likelihood를 구하면,
- $$E_z[ln p(x, z \mid \mu, \Sigma, \pi)] = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{nk} [{ln \pi_k + lnN(x_n \mid \mu_k, \Sigma_k)}]$$
  
  
**2.1. Variational GMM **  
**2.2. Dirichlet Process**
  
