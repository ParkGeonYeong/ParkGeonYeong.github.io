---
title : "Clustering with EM algorithm"
use_math : true
layout : single
classes : wide
---

Clustering은 Machine Learning에서도 unsupervised learning의 대표적인 갈래입니다. 
Unlabeled data를 군집화하고, 각 군집의 특성을 파악하는 과정입니다. 
이런 군집을 발견함으로써 우리는 각 군집에 관여하고 있는 latent factors가 어떤 차이를 보이고 있는지도 확인할 수 있습니다.
간단한 clustering 알고리즘 (K-means clustering)을 Expectation-Maximization algorithm(EM)의 관점에서
설명하겠습니다.  
  
**0. EM algorithm이란?**  
Expectation-Maximization 알고리즘(EM)은 우리의 parameter와 latent variable의 조절을 통해, 
Observation 데이터에 대한 Maximum likelihood solution을 찾아내는 것입니다. 
Parameter 뿐만 아니라 latent variable을 정의하고, 두 변수의 interaction을 고려해 
업데이트를 진행한다는 점이 타 알고리즘과의 차별점입니다. 우선 latent variable을 정의하는 과정이 필요하고, 이에 대한 
확률 분포를 얻어야 하기 때문에 다소 제약이 있지만, 생각보다 광범위한 모델에 적용 가능하다고 합니다. 
  
알고리즘은 어떤 latent variable z을 고려하는 것부터 시작합니다. 
기존의 log likelihood $$lnP(Xㅣ\theta)$$는 $$z$$의 marginalization으로 인해 $$ln(\sum_{z}P(X,Z|\theta))$$으로 표현되고 
이는 $$\sum$$이 로그 안에 들어있기 때문에 계산의 어려움이 있습니다.  
따라서 이를 밖으로 빼주는 과정이 필요한데 이 과정에서 로그 함수가 concave임을 활용합니다.  
  
우선 위의 식을 $$ln(\sum_{z}q(Z)\frac{P(x,Zㅣ\theta)}{q(Z)})$$  
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

**1. K-means Clustering**  
![image](https://user-images.githubusercontent.com/46081019/54875301-ba7a6380-4e3f-11e9-9fe1-066642621807.png)  
K-means Clustering은 K개의 centroid $$\mu_k$$을 parameter로 정의해 이를 유클리드 거리 손실에 대해 업데이트합니다.  
$$L = \sum_n\sum_kq(Z=k)||x_n-\mu_k||^2$$   
이 과정에서 latent variable $$Z=i$$는 nth data $$x_n$$이 어느 centroid에 할당되었는지를 의미합니다. 즉 
discrete variable에 대한 분포로써 $$x_n$$과 가장 가까운 $$\mu_k$$에 $$x_n$$을 할당하고, $$q(Z=k)=1$$이 됩니다.  
따라서 우선 \mu를 임의로 설정하고, 이를 기반으로 $$q(Z)$$를 설정한 다음, 다시 $$\mu_k$$에 대해 loss를 최적화합니다.  
이 과정에서 새로운 $$\mu_k$$를 계산하면, 곧 kth centroid에 할당된 데이터 평균이 됩니다. (증명 생략, $$\mu_k$$에 대해 편미분하면 유도 가능)  
  
K-means Clustering은 크게 두 가지의 단점을 갖고 있습니다. 우선 하이퍼 파라미터 k를 임의로 설정해야 하고, 
특정 데이터 $$x_n$$이 반드시 discrete하게 한 centroid에 assign되는 'Hard assign' 문제가 있습니다.  
각 문제는 Bayesian Non-parametric(Dirichlet process, ...)와 Gaussian Mixture Model(GMM)으로 해결 가능합니다. 
시간이 된다면 두 알고리즘에 대해 정리하도록 하겠습니다.
