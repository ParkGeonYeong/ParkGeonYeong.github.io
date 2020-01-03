---
title: Domain Adaptation Papers
use_math: true
classes: wide
layout: single
---

**개인적으로 읽은 Domain Adaptation 논문들을 정리합니다.**
- 디테일한 증명은 굉장히 중요한 Thm을 제외하고는 생략합니다. 빠르고 간략히 논문이 말하고자 하는 바를 요약하는데에 초점을 맞춥니다.   
- 결과 역시 굉장히 중요하지 않으면 생략합니다.  
- Deep Generative Model을 활용한 논문 / Theoretical한 논문 위주로 정리합니다.  
  
### Theoretical Analysis  
- [(2007) Ben-David et al., Analysis of Representations for Domain Adaptation](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)  
  - Domain Adaptation의 이론적 기반을 처음 닦은 논문. 이를 시작으로 동일 저자의 2010년 논문을 통해 domain adaptation의 큰 기틀이 완성
  - 기존의 Statistical Learning Theory에서 닦아 놓은 Empirical Risk Minimization 관련 inequality와 이론들은 하나의 True dataset과 여기서 sampling된 empirical sample 간의 차이에서 발생하는 오차를 확률적으로 규정하는 이론
  - 그러나 Dataset이 2개, 나아가 여러개일 경우에는 위의 이론을 직접 적용할 수 없음
  - 우선 dataset이 2개인 경우를 가정. 또, target data는 label을 갖고 있지 않아 직접적으로 empirical risk minimization을 할 수 없다고 가정. (Unsupervised)
  - 목표는 $$D_T$$에 대한 generalized error $$\epsilon_{T}(h)$$를 우리가 간접적으로 근사 가능한 $$\epsilon_{S}(h)$$으로 표현
  - **(Target Error) < (Performance of h on the Source Domain) + (Divergence btw induced $$\hat{D_S}$$, $$\hat{D_T}$$) + ...**
    - Source와 Target Domain Feature 간의 Divergence가 클 수록 classifier on top of feature는 쉽게 generalize되기 어려움
    - Example of Distance based on Norm: L1 distance (=Total Variance)
    - $$d_{L1}(D, D') = 2 sup_{B \in \mathbb{B}} \mid \mathbb{P}_{D}(B) - \mathbb{P}_{D'}(B) \mid$$
    - $$\mathbb{B}$$는 D, D'의 subset으로, real-valued distribution 간의 모든 subset에 대해서 sup을 취하는 것은 불가능
    - 이에 Divergence을 표현하기 위해 보조 Classifier h와 그 hypothesis space $$\cal{H}$$을 정의, **H-divergence** 도입
    - $$d_{\cal{H}}(D, D') = 2 sup_{h \in \cal{H}} \mid \mathbb{P}_{D}(I(h)) - \mathbb{P}_{D'}(I(h)) \mid $$
      - $$I(h) = {z \in \cal{Z}: h(z)=1}$$
    - Classifier h을 도입하고, 서로 다른 D, D'에 대한 I(h)를 subset으로 사용함으로써 finite sample을 이용한 divergence 계산이 가능해짐
    - 이후 Ganin et al., 등에서 이 h를 D, D'의 classifier으로 도입, adversarial trick을 사용하여 minmax problem을 풀게 됨 (max: sup(divergence), min: minimize upperbound w.r.t. feature extracter)
  - Bound 증명 scheme
    - $$\epsilon_T(h) = \mathbb{E}_{z \sim \hat{D_T}} [\mid f(z) - h(z) \mid]$$ 정의에서 시작, triangular inequality 등을 적극 활용
    - f: True labeling function
    - Optimal $$h^{*} \in \cal{H}$$을 정의하여 triangular ineq 사용
    - 이후 $$\mathbb{E}_{z \sim \hat{D_T}} [\mid h^{*}(z) - h(z) \mid]$$을 XOR operation을 사용하여 $$Pr_{\hat{D_T}}(z_{h^*} \triangle z_h)$$으로 표현
    - $$Pr_{\hat{D_T}}(z_{h^*} \triangle z_h) \leq \mid Pr_{\hat{D_S}}(z_{h^*} \triangle z_h) + Pr_{\hat{D_T}}(z_{h^*} \triangle z_h) \mid + Pr_{\hat{D_S}}(z_{h^*} \triangle z_h)$$ 트릭 사용
  
- [(2007) Blitzer et al., Learning Bounds for Domain Adaptation](https://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)  
  - 앞선 논문은 Unsupervised Domain Adaptation의 이론적 기반
  - 여기서는 Few-labeled target domain을 이용한 semi-supervised Domain Adaptation, 혹은 여러 개의 source domain을 활용한 multiple domain adaptation scheme 제공
  - 앞선 논문에서는 $$\epsilon_T(h)$$을 구하기 위해 $$\epsilon_T(h), \epsilon_S(h)$$ 그리고 $$\epsilon_S(h), \epsilon_S(\hat{h})$$ (Recall VC Generalization Bound)의 관계를 이용했다.
  - 여기서는 multiple source, 혹은 source and few-labeled target을 convex combination시켜서 $$\epsilon_{\alpha}(h)$$를 구한다.
  - 즉 **$$\epsilon_{T}(h), \epsilon_{\alpha}(\hat{h})$$**와 **$$\epsilon_{\alpha}(h), \epsilon_{\alpha}(\hat{h})$$**의 관계를 각각 구한 다음, 이를 alternating시키는 방식으로 upper-bound을 얻게 된다.
  - **1. Semi-Supervised DA** 
    - *Lemma 1.* (generalized alpha and target error)
      $$\epsilon_{\alpha}(\{h}) = \alpha \epsilon_{T}(h) + (1 - \alpha) \epsilon_{S}(h)$$이라 하자.   
      이 때, $$\begin{align*} 
      \mid \epsilon_{\alpha}(\hat{h}) - \eps_{T}(\hat{h}) \mid &= \mid (1 - \alpha) (\epsilon_{S}(h) - \epsilon_{T}(h) \\
      &= (1 - \alpha) \left\{ \mathbb{E}_{z \sim \hat{D_S}} [\mid f_S(z) - h(z) \mid] - \mathbb{E}_{z \sim \hat{D_T}} [\mid f_T(z) - h(z) \mid] \right\} \\
      &\leq (1 - \alpha) \left\{ \mathbb{E}_{z \sim \hat{D_S}} [\mid h^{*}_S(z) - h(z) \mid] - \mathbb{E}_{z \sim \hat{D_T}} [\mid h^{*}(z) - h(z) \mid] + \lambda_{S}^* + \lambda_{T}^* \right\} \\
      &\leq (1 - \alpha) (\frac{1}{2}d_{\cal{H} \triangle \cal{H}}(D_S, D_T) + \lambda)
      \end{align*} $$
    - *Lemma 2.* (generalized alpha and empirical alpha error)  
    Target domain은 $$\beta m$, Source domain은 $(1-\beta)m$$의 sample을 갖고 있다고 하자. 
    이때 $$ \begin{align*} 
    \hat{\epsilon_{\alpha}}({h}) &= \alpha \hat{\epsilon_{T}}({h}) + (1 - \alpha) \hat{\epsilon_{S}}({h}) \\
    &= \frac{(\alpha)}{\beta m} \sum{\mid h(z_i) - f_T(z_i) \mid} + \frac{(1-\alpha)}{(1-\beta)m} \sum{ \mid h(z_i) - f_S(z_i) \mid} \\ 
    &= \frac{1}{m} \sum{F(z_i)} \end{align*}$$으로 쓸 수 있으며, $$F(z_i)$$는 $$i < \beta m$$일 경우 $$\frac{(\alpha)}{\beta m} {\mid h(z_i) - f_T(z_i) \mid}$$, $$i > \beta m$$일 경우 $$\frac{(1-\alpha)}{(1-\beta) m} {\mid h(z_i) - f_S(z_i) \mid}$$이다.  
    분류 에러는 1을 넘을 수 없으므로, $$F(z_i)$$는 i에 따라 $$\frac{\alpha}{\beta}$$, 혹은 $$\frac{(1-\alpha)}{(1-\beta) m}$$에 upperbounded이다.  
    이 때 $$\mathbb{\hat{\epsilon_{\alpha}}({h})} = \epsilon_{\alpha}({h})$$이므로,   
    $$Pr(\mid \hat{\epsilon_{\alpha}}({h}) - \epsilon_{\alpha}({h}) \mid \leq \eps)$$에 Hoeffding's inequality를 적용할 수 있다.   
      - Hoeffding's inequality :   
      ![image](https://user-images.githubusercontent.com/46081019/71712906-ae014d80-2e4a-11ea-8940-f2bae55ab02c.png)  
        - 직관적으로, convex-combined source dataset 역시 어쨌든 하나의 단일 dataset이기 때문에 generalization bound을 구할 수 있을 것이다. 다만 일반적인 하나의 dataset과는 차이가 조금 있기 때문에 hoeffding's inequality를 사용하기 전에 convex combined data을 하나의 시그마에 통합하는 트릭을 통해 empirical mean error를 우선 정의하고 넘어가는 과정이 필요했다.   
    - $$\mid \hat{\epsilon_{\alpha}}({h}) - \epsilon_{\alpha}({h}) \mid$$와 $$\mid \epsilon_{\alpha}({h}) - \epsilon_{T}({h}) \mid$$의 inequality를 얻었으므로 이를 종합하여 $$\epsilon_T(\hat{h})$$에 대한 generalization bound을 얻을 수 있다.  
    - ![image](https://user-images.githubusercontent.com/46081019/71712996-151f0200-2e4b-11ea-8e7e-07d8063833b0.png)  
    - alpha가 0일 경우 Unsupervised DA, alpha가 1일 경우 Target domain만을 활용한 generalization bound을 얻을 수 있다.
    - Optimal alpha는 그 중간에 있는데, 이는 beta 값을 고려하여 결정된다. 즉 beta(from target domain)이 클 수록 target labeled data가 많기 때문에 alpha을 높게 잡는게 유리하다. 
    
  - **2. Multi-source DA** 
    - Multi-source DA 역시 거의 비슷한 방식으로 증명할 수 있다.  
    - ![image](https://user-images.githubusercontent.com/46081019/71713090-86f74b80-2e4b-11ea-912a-1a0caa0521b7.png)  
    - 이때 H-divergence가 Target과 convex-combined source domain에 대해 정의되어 있다. 이때 만약 더 관련성이 높은 source domain에 높은 convex weight가 가해졌다면 target과의 H-divergence를 줄이기 더 수월할 것이다. 아래 그림 참고
    - ![image](https://user-images.githubusercontent.com/46081019/71713129-c0c85200-2e4b-11ea-8d6e-38866d36848a.png)  
 
  
- [(2019) Zhao et al., On Learning Invariant Representations for Domain Adaptation](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf)  
  
### Combined with Deep Generative Model    
- [(2017) Sankaranarayanan et al., Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/abs/1704.01705)  
  
- [(2017) Saito et al., Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560)  
  
- [(2018) Liu et al., A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation](https://papers.nips.cc/paper/7525-a-unified-feature-disentangler-for-multi-domain-image-translation-and-manipulation.pdf)  
  
- [(2018) Hoffman et al., CyCADA: Cycle-Consistent Adversarial Domain Adaptation](https://arxiv.org/abs/1711.03213) 
  
  
### Disentangling Domain Information  
- [(2016) Bousmalis et al., Domain Separation Networks](https://arxiv.org/abs/1608.06019)  

- [(2019) Peng et al., Domain Agnostic Learning with Disentangled Representations](https://arxiv.org/abs/1904.12347)  
  
  
### Domain Generalization / Multi-Source Adaptation  
- [(2018) Li et al., Domain Generalization with Adversarial Feature Learning](https://www.ntu.edu.sg/home/sinnopan/publications/[CVPR18]Domain%20Generalization%20with%20Adversarial%20Feature%20Learning.pdf)  
  
- [(2018) Gong et al., DLOW: Domain Flow for Adaptation and Generalization](https://arxiv.org/abs/1812.05418)  
  
- [(2018) Li et al., Extracting Relationships by Multi-Domain Matching](https://papers.nips.cc/paper/7913-extracting-relationships-by-multi-domain-matching.pdf)  
  
- [(2018) Zhao et al., Adversarial Multiple Source Domain Adaptation](https://papers.nips.cc/paper/8075-adversarial-multiple-source-domain-adaptation.pdf)  

  
### Suggest New Loss  
- [(2017) Haeusser et al., Associative Domain Adaptation](https://arxiv.org/abs/1708.00938)  
