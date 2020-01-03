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
  - 목표는 $D_T$에 대한 generalized error $\eps_{T}(h)$를 우리가 간접적으로 근사 가능한 $\eps_{S}(h)$으로 표현
  - **(Target Error) < (Performance of h on the Source Domain) + (Divergence btw induced $\hat{D_S}$, $\hat{D_T}$) + ...**
    - Source와 Target Domain Feature 간의 Divergence가 클 수록 classifier on top of feature는 쉽게 generalize되기 어려움
    - Example of Distance based on Norm: L1 distance (=Total Variance)
    - $d_{L1}(D, D') = 2 sup_{B \in \mathbb{B}} \mid \mathbb{P}_{D}(B) - \mathbb{P}_{D'}(B) \mid$
    - $\mathbb{B}$는 D, D'의 subset으로, real-valued distribution 간의 모든 subset에 대해서 sup을 취하는 것은 불가능
    - 이에 Divergence을 표현하기 위해 보조 Classifier h와 그 hypothesis space $\cal{H}$을 정의, *H-divergence* 도입
    - $d_{\cal{H}}(D, D') = 2 sup_{h \in \cal{H}} \mid \mathbb{P}_{D}(I(h)) - \mathbb{P}_{D'}(I(h)) \mid
      - $I(h) = {z \in \cal{Z}: h(z)=1}$
    - Classifier h을 도입하고, 서로 다른 D, D'에 대한 I(h)를 subset으로 사용함으로써 finite sample을 이용한 divergence 계산이 가능해짐
    - 이후 Ganin et al., 등에서 이 h를 D, D'의 classifier으로 도입, adversarial trick을 사용하여 minmax problem을 풀게 됨 (max: sup(divergence), min: minimize upperbound w.r.t. feature extracter)
  - Bound 증명 scheme
    - $\eps_T(h) = \mathbb{E}_{z \sim \hat{D_T}} [\mid f(z) - h(z) \mid]$ 정의에서 시작, triangular inequality 등을 적극 활용
    - f: True labeling function
    - Optimal $h^{*} \in \cal{H}$을 정의하여 triangular ineq 사용
    - 이후 $\mathbb{E}_{z \sim \hat{D_T}} [\mid h^{*}(z) - h(z) \mid]$을 XOR operation을 사용하여 $Pr_{\hat{D_T}}(z_{h^*} \triangle z_h)$으로 표현
    - $Pr_{\hat{D_T}}(z_{h^*} \triangle z_h) \leq \mid Pr_{\hat{D_S}}(z_{h^*} \triangle z_h) + Pr_{\hat{D_T}}(z_{h^*} \triangle z_h) \mid + Pr_{\hat{D_S}}(z_{h^*} \triangle z_h) 트릭 사용
  
- [(2007) Blitzer et al., Learning Bounds for Domain Adaptation](https://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)  
  - 
- [(2019) Zhao et al., On Learning Invariant Representations for Domain Adaptation](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf)  
  
### Combined with Deep Generative Model    
- [(2017) Sankaranarayanan et al., Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/abs/1704.01705)  
  
- [(2017) Saito et al., Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/abs/1712.02560)  
  
- [(2018) Liu et al., A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation](https://papers.nips.cc/paper/7525-a-unified-feature-disentangler-for-multi-domain-image-translation-and-manipulation.pdf)  
  
- [(2019) Peng et al., Domain Agnostic Learning with Disentangled Representations](https://arxiv.org/abs/1904.12347)  
  
  
### Domain Generalization / Multi-Source Adaptation  
- [(2018) Li et al., Domain Generalization with Adversarial Feature Learning](https://www.ntu.edu.sg/home/sinnopan/publications/[CVPR18]Domain%20Generalization%20with%20Adversarial%20Feature%20Learning.pdf)  
  
- [(2018) Gong et al., DLOW: Domain Flow for Adaptation and Generalization](https://arxiv.org/abs/1812.05418)  
  
- [(2018) Li et al., Extracting Relationships by Multi-Domain Matching](https://papers.nips.cc/paper/7913-extracting-relationships-by-multi-domain-matching.pdf)  
  
- [(2018) Zhao et al., Adversarial Multiple Source Domain Adaptation](https://papers.nips.cc/paper/8075-adversarial-multiple-source-domain-adaptation.pdf)  

  
### Suggest New Loss  
- [(2017) Haeusser et al., Associative Domain Adaptation](https://arxiv.org/abs/1708.00938)  
