---
title: "Manifold Learning 알고리즘 (t-SNE,...)"
use_math: true
classes: wide
layout: single
---

[이전 포스트](https://parkgeonyeong.github.io/Linear-Methods-in-ML/)에서 
Low-dimensional representation을 얻어 학습하는 PCA, LDA, CCA 등을 다뤘었다. 
Eigen decomposition method를 기반으로 데이터를 새로운 축에 '선형으로' orthogonal projection(사영)시키는 과정이 핵심이었다. 
여기서는 projection보다는 데이터 자체의 inherent non-linear manifold를 배우는 알고리즘들을 알아본다. 
  
Manifold의 학습은 곧 다른 의미로 kernel 테크닉의 예시라고 할 수 있다. 
Kernel이 given data를 High-dimensional Reproducing Kernel Hilbert Space에서 갖고 논다면, 
Manifold를 배우는 것은 그 반대로 dimension을 효과적으로 줄이는 과정이다. 
  
고차원의 데이터(이미지 등)은 그 자체로는 다루기 까다롭다. 가령 일반적으로 사용되는 euclidean distance도 차원에 값 영향을 많이 받는다.
하지만 대부분의 고차원 데이터는 그 안에 manifold를 가지기 마련인데, 가령 object의 성격이 제한될 때(e.g., 특정 강아지 사진)
데이터 내부적으로 dependent한 correlation이 생기기 때문이다. (e.g., 강아지의 눈, 코, 귀 위치는 어느 정도 예측 가능한 범위에 있다) 
즉 이러한 데이터의 고유한 구조를 impact하게 학습하고, 이를 visualize할 수 있다면 데이터의 이해에 도움이 된다. 
  
많은 Manifold learning은 고차원에서의 데이터간 관계(usually dependent to distance norm, ...)를 저차원에서 효율적으로 유지하는 것을 목표로 한다. 
이러한 관점에서 가장 기초적인 알고리즘인 Isomap, Locally linear embedding(LLE), 
그리고 아직까지 많이 활용되는 Stochastic neighbor embedding(SNE), t-SNE까지 알아본다.  

**0. Isometric feature mapping(Isomap)**  
Isomap은 가장 오래되고 기초적인 manifold learning 알고리즘 중 하나이다. 데이터 간의 linear distance를 기반으로 graph representation을 학습한다. 
과정은 다음과 같다.  
- 각 data point별로 L(hyper-parameter) nearest neighbor를 찾고, 이를 그래프로 잇는다. 
- 그래프의 edge에 euclidean distance를 기반으로 가중치를 매긴다. 
- i-th sample에서 j-th sample 간의 거리인 $$d_{ij}$$를 잰다. (By Dijkstra or other) 이를 기반으로 distance matrix $$D$$를 구한다.
- Gram matrix $$G=-\frac{1}{2}HDH$$를 구한다. ($$H=I-\frac{1}{N}11^{T}$$) 
- G를 eigen-decompose ($$v, \lambda$$)한다.
- top K eigen pairs를 구해 사영시킨다.   
  
Isomap은 각 데이터 별로 충분히 가까운 데이터들에 대한 인접 그래프를 얻어 매니폴드를 표현하는 과정이다. 
이때 두 데이터(노드) 간의 최단 경로를 euclidean distance를 기반으로 구하며, 이는 곧 high-dimensional space에서도 
충분히 local한 영역에 대해서는 euclidean space를 합당하게 가정할 수 있기 때문이다. 
  
![i1-4](https://user-images.githubusercontent.com/46081019/58704022-90b45e80-83e5-11e9-92ad-6157137ef982.png)  
  
**1. Locally Linear Embedding(LLE)**  
LLE 역시 geometric intuition에 기반한 linear model를 이용해 manifold를 배운다. 
High-dimensional data는 Global한 관점에서 non-linear하지만, 이를 우선 무시하고 
Isomap과 비슷하게 각 데이터의 이웃들이 locally linear하다고 해보자. 
원 데이터 $$D=[x_1, x_2, ..., x_N], x\in R^d$$을 $$[z_1, z_2, ..., z_N], z\in R^K$$으로 보내는 것이 목적이다.  
  
![다운로드 (1)](https://user-images.githubusercontent.com/46081019/58705042-edfddf00-83e8-11e9-8323-7886cd428e4f.png)  
  
과정은 다음과 같다. 
- Isomap과 동일하게 각 data point별로 L(hyper-parameter) nearest neighbor set $$\xi$$를 찾는다
- $$\xi$$의 모든 data point에 임의에 weight를 부과한다. 이후 Least-square 문제를 풀어 weight를 구한다.
  - $$W = argmin_{W}\sum_{i=1}^N \parallel x_i -\sum_{j \in \xi}W_{ji}x_j \parallel^2, (\sum_j w_{ji}=1)$$ 
  - 괄호 안의 식을 constraint으로 고려하여 lagrangian method를 사용하면 w를 구할 수 있다. 
- High dimensional space에서 구한 weight W를 이용해 low dimensional projection $$z$$을 구한다.
  - 이를 low-dimensional coordinate으로 mapping한다고 한다.
  - High-dimension space에서 구한 가중치가 각 data point의 '지역적' 관계에 대한 정보를 갖고 있기 때문에, 이를 최대한 유지한 채 이번에는
  z에 대해서 푸는 것이다.
  - $$Z=argmin_{z}\sum_{i=1}^N \parallel z_i -\sum_{j \in \xi}W_{ji}z_j \parallel^2$$
  - 제약식은 $$\sum{z_i}=0, \frac{1}{N}ZZ^T=I$$
  - Off-diagonal covariance를 가정한다.
  
**2. Stochastic Neighbor Embedding(SNE)**  
LLE가 neighbor 및 weight assignment를 'hard'하게 했다면, SNE는 'soft'하다. 
즉 "neighborhood" 그 자체를 stochastic하게 정의하는 것이다. 
High dimensional space에서 어떤 data point와의 거리가 가까울 수록 해당 point와의 관계를 tight하게 유지하며, 
이는 딱 잘라 neighbor를 정의했던 LLE와 다른 점이다.
  
가령 i-th data point 관점에서 j-th point에 대한 neighborhood 혹은 영향력은 조건부 확률 형식을 빌려, 
$$p_{j \mid i}$$으로 표현한다. 이를 gaussian distribution 형식으로 표현하면 다음과 같다. 
  
![image](https://user-images.githubusercontent.com/46081019/58705689-e2131c80-83ea-11e9-96d7-630f38e6c5b7.png)   
  
- 이때 $$\sigma_i$$는 각 데이터에 대해 개별적인 hyper-parameter이다. 
- Softmax의 temperature parameter와 비슷한 역할을 하는데, $$sigma$$가 작을 수록 두 데이터 포인트의 차이는 보다 부각되며 local한 영향력이 커진다. 
- 반면 $$sigma$$가 작다면 보다 넓은 범위의 point들이 영향력을 미치게 된다. 
  SNE에서 가장 중요한 hyper-parameter라고 할 수 있으며 데이터가 너무 작은 경우 sigma를 줄여야 한다. 
    
위와 동일한 논리를 low dimensional space의 vector $$y$$에 적용할 수 있다.  
$$q_{j \mid i} = \frac{exp(\parallel y_i -y_j \parallel^2)}{\sum_k exp(\parallel y_i -y_k \parallel^2)}$$
  
이렇게 i-th data point에 대해 high, low-dimensional neighborhood probabilistic distribution $$P_i, Q_i$$를 얻었다. 
SNE의 핵심은 서로 다른 두 space에서 구한 distribution이 크게 차이가 나지 않도록 만든다는 것이다. 
- Cost $$C=\sum_i KL(P_i \mid\mid Q_i)=\sum_i\sum_j p_{j \mid i}log(\frac{p_{j \mid i}}{q_{j \mid i}})$$
- 각 embedded new data points $$y_i$$에 대해 cost를 편미분하여, $$\triangledown{y_i}$$을 구한다.  
    
t-SNE는 SNE와 본질적으로 크게 다르지 않은 알고리즘이다.
SNE에서는 conditional probability로 (i, j) 관계를 표현하였으며 이는 진정한 distance metric이라 할 수 없는데, asymmetric하기 때문이다. 
따라서 이를 보완하고, SNE의 gaussian 대신 t-distribution을 사용한 것이 t-SNE이다. 
즉 cost $$C=\sum_i KL(P_i \mid\mid Q_i)=\sum_i\sum_j p_{ij}log(\frac{p_{ij}}{q_{ij}})$$으로, conditional probability 형식이 아니다.
또한 student t-distribution을 사용하는데, 이는 t-distribution이 gaussian 대비 heavy tail을 갖기 때문에 high dimension space에서의 
large distance를 반영하기 더 유리하기 때문이다. 
  
df=1일 때 t-SNE에서 neighborhood representation $$q_{ij}$$는 다음과 같다.  
$$q_ij = \frac{(1 + \parallel y_i - y_j \parallel^2)^{-1}}{\sum_{k,l} (1 + \parallel y_k - y_k \parallel^2)^{-1}}$$  
이는 t-distribution의 chi-squared 항이 반영된 것으로 보인다. 1이 붙은 이유는 역수를 취하는 과정에서 inf가 나오지 않도록 추가한 것이다. 
직관적으로는 (i, j)가 가까울 수록 확률적으로 두 data point는 가까운 state이며, 따라서 $$y_i, y_j$$가 크게 변하지 않을 가능성이 높다.  
MNIST에 적용한 예시는 다음과 같다.   
  
![visualizing-data-using-tsne-53-638](https://user-images.githubusercontent.com/46081019/58706879-2d7afa00-83ee-11e9-836a-c20f1d7ca27f.jpg)
  

