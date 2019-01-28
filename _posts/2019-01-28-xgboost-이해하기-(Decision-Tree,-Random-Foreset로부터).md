---
title: "xgboost 이해하기 (Decision Tree, Random Foreset로부터)"
use_math: True
layout: single
classes: wide
---

본 글은 다음 자료를 주로 참고했습니다.
*xgboost*  
- XGBoost: A Scalable Tree Boosting System, T Chen et al.
- [Complete Guide to Parameter Tuning in xgboost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
*Random Forest, Decision Tree*  
- [Random Forest](https://link.springer.com/article/10.1023/A:1010933404324)
- [CMU Docu](https://www.cs.cmu.edu/~ggordon/780-fall07/fall06/homework/15780f06-hw4sol.pdf)
- [Conditional Entropy](https://en.wikipedia.org/wiki/Conditional_entropy)
- [NYU lecture](file:///C:/Users/박건영/Desktop/lecture11.pdf)

**0. Decision Tree란 무엇인가?**  
Decision Tree는 트리 기반 기법으로, 주어진 데이터 (Y, x1, x2, ..., xN)을 class $$C_i$$로 판정할 수 있는 의사 결정 모델입니다.
이때 tree의 branch는 주어진 feature x1, ..., xN 중에서 하나 또는 여러 attribute를 선택하게 됩니다. 완성된 Tree에 분류 대상 input x를 주면 
적절한 branch를 따라가 최종 node의 class를 할당받게 됩니다. 
Decision tree는 non-parametric(Underlying data distribution에 대해 parametrized distribution을 가정하지 않음), Supervised(Given label y) 모델입니다.  

Decision Tree의 핵심은 "어떻게 branch를 나눌 것인가?"입니다. 
Feature가 많아질 수록 각 branch마다 가능한 attribute(Hypothesis space)가 매우 다양하기 때문입니다. 
Decision Tree에서는(그리고 여러 파생 모델은) 이 문제를 Greedy-recursive하게 풀게 됩니다. 
즉, Empty tree에서 시작해, 나눌 수 있는 경우의 수를 greedy하게 선택합니다. 이때 일종의 목적함수로 작용하는게 **Information-Gain**입니다.  
좋은 Decision tree란 곧 각 leaf의 uncertainty가 낮으며 predictable한 구조를 의미합니다. (leaf의 class 분포가 확실할수록 uncertainty는 낮다) 
**즉 decision tree는 각 class의 uncertainty 분포가 고른 상태(=High entropy)에서 특정 class의 uncertainty가 낮은 상태(=low entropy)로 전환되어야 합니다.**
Information gain은 Decision tree의 'spliting'으로 인해 entropy 이득을 얼마나 볼 수 있는지를 나타냅니다.  
$$IG(X) = H(Y) - H(Y|X)$$  
이때 IG(X)는 X를 이용한 splitting이 발생시키는 information gain, H(Y)는 splitting 이전의 entropy, 
H(Y|X)는 splitting 이후에 X의 조건부 확률로 Y의 분포를 설명하는 Conditional entropy입니다. 
즉 decision tree는 현재 취할 수 있는 여러 splitting 경우의 수 중에서 가장 $$IG(X_i)$$가 높은 $$X_i$$를 이용해 splitting합니다.
(이때 Information gain은 cross-entropy 외에도 Gini index 등을 통해서 정의 가능합니다.)  

**다만 Information Gain은 어떤 X에 대해서도 non-negative하기 때문에, 
제한을 걸지 않고 트리를 계속 키울 경우 지나치게 많은 splitting으로 인해 over-fitting이 발생할 수 있습니다.** 
이를 위해 실제 tree에서는 pruning을 통해 size을 제한합니다. 가령 Continuous한 variable의 경우는 MSE와 같은 objective function에 tree size T를 
regularizer로 더할 수 있습니다.  

**1. Random Forest란 무엇인가?**  
Random Forest는 'Forest'에서 알 수 있듯이 여러 tree predictor를 합친(ensemble, bagging) 모델입니다.  
(Random Forest = Set of trees, each of which sees a different subsets of data and features)  
앞서 확인했듯이 Decision tree는 bias error는 없지만(training error를 0으로 만들 수 있음), over-fitting이 되기 쉽습니다. 
즉 variance가 높아질 수 있다는 단점이 있습니다.
이 high-variance를 여러 independent한 tree로 낮추는 접근 방식이 Random Forest입니다. (이를 bagging(bootstrapping+aggregation) 방식이라고 합니다.)
Independenet한 tree들이 각각 prediction에 참여하고, 이를 평균내면 통계적으로 variance는 자연히 낮아지기 때문입니다. 
최종 결과는 continuous output인 경우 각 tree output의 average를, categorical output인 경우 voting을 활용합니다. 
이때 각 tree는 feature와 data를 랜덤하게 선택합니다. 
Feature의 경우 전체 feature set에서 random하게 feature을 선택하거나, selected feature를 다시 linear하게 조합해서 사용합니다. 
데이터 역시 동일한 방식으로 bootstrap datasets에서 random하게 선택합니다. 
이때 확률적으로 bootstrapping 과정에서 남은 데이터를 test set으로 활용해 error를 추산할 수 있습니다.
이를 out-of-bag error estimation이라고 합니다.   
  
**Random Forest가 좋은 앙상블 모델인 이유는 variance를 줄이면서도 unbiased error를 갖기 때문입니다.** 
**구체적으로는, forest의 tree 개수가 많아질수록 각 tree의 prediction에서 발생한 prediction error의 기댓값이 전체 feature set을 이용한 prediction error로
수렴합니다.** Bias 측면에서 전체 tree와 이론적으로는 큰 차이가 없다는 뜻입니다. 
이로 인해 Random Forest는 low-bias와 low-variance 이득을 동시에 챙길 수 있게 됩니다.  
$$Proof$$  
위를 증명하기 위해서는 결국 각 tree의 y=j에 대한 예측의 기댓값이 전체 fixed tree $$\theta$$에 대한 기댓값으로 수렴한다는 것을 보이면 됩니다.  
$h_{k}(x) = h(x|\theta_k)$를 subset k에 대한 individual tree라고 합니다. 
이때 주어진 x에 대해서 N개의 tree가 평균적으로 class j를 판정할 확률은 
$$\frac{1}{N}\sum_{k=1}^{N}I(h(\theta_k,x)=j)$$입니다.  
이때 전체 $$\theta$$가 총 K개의 hyper-rectangles $$S_1, ..., S_k$$을 형성하고 있다고 합시다. 
주어진 x에 대해 j를 판정할 확률은, "어떤 hyper-rectangle $$S_k$$에 속한 x에 대해 tree들이 평균적으로 j를 판정한 사건"을 모든 rectangles에 대해 평균내는 것과 같습니다.  
\frac{1}{N}\sum_{k=1}^{N}I(h(\theta_k,x)=j) = \frac{1}{N}\sum_{k}N_kI(x{\in}S_k)   
이때 $$N_k = \frac{1}{N}\sum_{m=1}^{N}I(\phi(\theta_m)=k)$$으로, 전체 tree에 대해서, j로 판정받은 데이터 x가 $$S_k$$에 속할 기댓값입니다.  
($$\phi(\theta_m)=k$$는 [{x: h(\theta_m, x)=j}] = S_k)  
이때 tree의 개수가 매우 많아질 경우 $$\theta_m$$의 예측을 모두 평균낸 $$N_k$$는, 곧 "전체 셋인 $$\theta$$가 j로 판정한 x가, $$S_k$$에 속할 확률"으로 수렴합니다. 
(수많은 $$theta_m$$의 의견이 모여 전체 $$\theta$$의 의견을 대변한다고 해석할 수 있습니다. Law of Large Number가 사용되었습니다.)  

