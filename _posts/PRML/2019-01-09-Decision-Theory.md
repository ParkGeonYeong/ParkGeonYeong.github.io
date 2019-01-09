---
title: "1.5 Decision Theory"
use_math: true
---

본 글은 PRML 1.5 Decision Theory를 QnA 형식으로 정리했습니다.  
<http://norman3.github.io/prml/docs/chapter01/5>을 참고했습니다.  
의, 오역 및 잘못된 의견이 첨가되었을 수 있습니다.  
**0. Probability**
Bayesian statistics에서 probability는 uncertainty의 inference입니다.

**1. Decision Theory가 무엇인가?** 
- Misclassification rate, 혹은 *Expected loss*를 최소화하는 Decision boundary를 찾는 것입니다.
  - 여기서 misclassification rate는 arbitrary given x에 대해 ground truth의 category를 얼마나 잘 찾아내는지를 의미합니다. 
  >$$p(correct) = \sum_{k=1}^{K}\int_{R_k}^{}p(x, C_k)dx  \propto \sum_{k=1}^{K}\int_{R_k}^{}p(C_k|x)p(x)dx$$  
  - $p(x)$를 redundant하다 가정하면, 위 수식에서 결국 $p(C_k|x)$의 값을 최대화하는 방향으로 $kth category$를 결정해야 합니다. 
  
  - 반면 Expected loss는 specific given x의 decision loss에 대해서는 penalty를 부과하는 function입니다.
    - 위 수식에서는 모든 ground truth category 종류를 균등하게 취급했지만,
    - 만약 특정 category로 잘못 판정됬을때 큰 페널티를 부과하려 한다면 별도의 Loss function을 정의해야 합니다.
    - 우리의 목표는 이러한 expected total loss를 최소화하는 $R_j$ decision region(~=boundary)를 정의하는 것입니다.
  - 만약 모든 category 선택지에 대해 $p(C_k|x)$가 고만고만하다면
    - 즉, 모든 category에 대해 비슷하게 불확실하다면
    - 해당 given x에 대한 선택을 보류하는게 유리합니다.
    - 이를 reject option이라고 합니다.
    
**2. Inference와 decision의 차이가 무엇인가?**
- Inference는 주어진 training dataset으로 $p(C_k|x)$에 대한 모델을 학습하는 과정. Decision은 이를 활용해 주어진 data에 class를 assign하는 과정
- 결국 목표는 decision이고, 이 문제를 푸는 방법은 3가지가 있습니다.
  (1) Generative model
  - Generative model은 데이터의 클래스-조건부 밀도와 prior probability를 각각 추론해서 posterior probability를 얻어냅니다. ~~사실상 bayesian~~
  - 얻어낸 $p(C_k|x)$로 새로운 new train data sample point를 유추할 수 있어서 generative라는 이름이 붙었습니다.
  - 이는 Bayesian statistics에 근거합니다.
  - $p(C_k|x) = \frac{p(x|C_k)p(C_k)}{p(x)}$
  - 이때 주어진 $C_k$ category들의 x로부터 각각 $p(x|C_k)$를 얻어내고, prior class probability $p(C_k)$까지 얻어냅니다.
  - 이를 통해 posterior probability를 얻어냅니다.
  - 분포를 추론했기 때문에 임의로 새 데이터 생성도 가능합니다.
  - 가령 Linear Discriminant Analysis(LDA)는 각 카테고리의 데이터 분포가 정규 분포라고 가정하여 $p(x|C_k)$를 얻어내고, 카테고리별 분포로부터 $p(C_k)$를 얻어내서 posterior probability를 계산합니다. 마지막으로 (1)에서의 decision theory를 통해 x에 class를 assign합니다.
  - ~~LDA가 왜 Linear Generative Analyis라고 불리지 않는지 모르겠습니다~~  
  (2) Discriminant model
  - Discriminant function은 posterior probability를 바로 추론합니다.
  - 흔히 아는 SVM, tree 등이 포함됩니다.  
  
  *두 모델링의 차이점*
  ![image](https://user-images.githubusercontent.com/46081019/50903340-e0a77e80-1460-11e9-834f-1d3eaf284476.png)
  상황 : x=0.8에 대해 decision을 내려야 하는 상황
  왼쪽(Generative) : 파랑, 빨강 class probability $p(C_b), p(C_r)$는 대충 비슷하고... 기존 데이터를 보아하니, 클래스 밀도가 빨강이 훨씬 높으니까 빨강으로 결정해야겠다!   
  오른쪽(Discriminant) : 클래스 밀도는 모르겠지만... 사후 확률을 학습한 바로는 빨강일 확률이 높으니까 빨강으로 결정!   
  
  
  (3) Discriminant Function
  - Discriminant model은 사후 확률 모델을 만들지만, discriminant function은 확률을 따지지 않고 바로 input을 결정해 버림
    
