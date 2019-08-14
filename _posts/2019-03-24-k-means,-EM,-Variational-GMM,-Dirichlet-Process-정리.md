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
  - $$likelihood \\ 
  = ln(\sum_{z} P(x,Z|\theta)) \\ 
  = ln(\sum_{z}q(Z)\frac{P(x,Zㅣ\theta)}{q(Z)})$$  
  - $$P(x, z)$$에 대해 쓴 꼴은 likelihood의 complete 꼴이라 부름 (1.2.2 참고)
으로 변형합니다. $$q(Z)$$는 latent variable에 대한 확률 분포입니다.
Jensen's inequality에 따라 $$ln(\sum_{z}q(Z)\frac{P(x,Zㅣ\theta)}{q(Z)})\geq\sum_{z}q(Z)ln(\frac{P(x,Zㅣ\theta)}{q(Z)}) $$
이 됩니다. 우변의 분모, 분자를 분리하면 $$\sum_{z}q(Z)ln(P(x,Zㅣ\theta))-\sum_{z}q(Z)ln(q(Z))$$으로, 
$$E_{q(Z)}ln(P(x,Z|\theta))+H(q)$$와 동일한 식이 됩니다. 따라서 jensen's inequality를 통해 log likelihood의 lower bound을 찾았습니다. 
  
또 다른 방식으로 lower bound을 찾아보면 다음과 같습니다. 
$$ln(\sum_{z} P(x,Z|\theta))\\
= ln(\sum_{z} P(Z|x,\theta)P(x|\theta))\\
= ln(\sum_{z}q(Z)\frac{P(Z|x,\theta)P(x|\theta)}{q(Z)})\\
= ln(\sum_{z}q(Z)\frac{P(Z|x,\theta)}{q(Z)})+ln(\sum_{z}q(Z)P(x|\theta))\\
= ln(P(x|\theta))-KL(q(Z)||P(Z|x,\theta))$$  
  
즉 원래 log-likelihood의 lower-bound를 tight시킬 조건은
latent variable의 distribution $$q(Z)$$가 $$P(Z|x, \theta)$$와 일치할 때입니다.  
따라서 완벽히 학습되지 않은 parameter $$\theta$$를 기반으로 우선 $$q(Z)$$를 구하고, 
다시 q(Z)를 통해 parameter $$\theta$$를 업데이트합니다. 
이러한 두 변수 $$q(Z), \theta$$의 interaction을 통해 log-likelihood의 Lower Bound을 꾸준히 maximize할 수 있습니다.
  
처음 구한 lower bound 식을 $$Q(\theta, q)$$, KL-divergence를 포함하고 있는 식을 $$L(\theta|q)$$라고 합시다.  
이때 슈도-알고리즘 순서는 다음과 같습니다.  

1) time=0에서 $$\theta^0$$ 초기화  
2) $$q^0(Z) = P(Z|x, \theta^0)$$, $$L(\theta|q)$$ is tighten  
3) $$\theta^1 = argmax_{\theta}E_{q^0(Z)}ln(P(x,Z|\theta^0)$$, $$Q(\theta, q)$$ is tighten  
  
이를 likelihood가 수렴할 때까지 time-step t에 대해 반복합니다.
**1.2.2 EM for GMM**  
  
**1.2.3 Generalized EM**  
  
