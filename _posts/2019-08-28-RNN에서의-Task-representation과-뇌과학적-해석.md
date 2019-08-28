---
title: "Recurrent Meta-Learning과 뇌과학적 해석"
use_math: true
layout:wide
classes: single
---  
  
  
본 자료는 다음을 참고했습니다.  
- [Yang et al., "Task representations in neural networks trained to perform many cognitive tasks", Nature Neuro.](https://www.nature.com/articles/s41593-018-0310-2#Sec32)  
- [Wang et al., "Prefrontal cortex as a meta-reinforcement learning system", Nature Neuro.](https://www.nature.com/articles/s41593-018-0147-8.pdf?source=post_page---------------------------)  
  
  
메타 러닝은 기존의 neuroscience의 실험론으로 쉽게 접근하기 어려운 문제이다. 
Neuro 연구의 가장 큰 두 가지 방법론이 optogenetics 등을 활용한 animal studying, fMRI 등 imaging을 활용한 human studying인데 
전자는 multi-task를 훈련시키기 어렵고, 후자는 resolution 문제로 디테일한 neural mechanism을 확인하기 어렵다.  
이에 Computational Neuroscientist를 필두로 동물의 메타 러닝을 wet lab 실험이 아닌 modeling으로 풀고자 하는 움직임이 있다.   
[Learning to reinforce Learn](https://arxiv.org/abs/1611.05763)을 시작으로 딥마인드 뉴로 연구 조직에서 
Prefrontal Cortex-Subcortical circuit을 메타 러닝의 원리로 처음 제안하였다. 이 때문인지 동물의 메타 러닝 연구는 메모리와 결합된 
recurrent neural network을 가장 많이 활용하는 것 같다.   
여기서는 메타 러닝에서도 구체적으로 task representation이 어떻게 일어나는지를 recurrent modeling으로 보인  
"Task representations in neural networks trained to perform many cognitive tasks", *Nature Neuro. 2018*을 figure 위주로 리뷰한다.   
  
  
Abstract가 간결하게 잘 씌여 있는데, 이를 번역하면 다음과 같다.  
> "Brain은 여러 task에 대해 유연하게 적응할 수 있지만, 그 기저의 원리는 기존 실험 및 모델링적 방법론으로는 쉽게 밝혀지지 않았다. 
우리는 워킹 메모리, 의사 결정, 분류, 억제 등의 특성을 지닌 20개의 서로 다른 인지 문제를 네트워크에 학습시켰다. 
학습 결과, RNN의 각 은닉 노드들이 클러스터를 이뤄 이러한 서로 다른 인지 제어 과정에 기능적으로 특화된다는 점을 발견했다. 
이에 우리는 간단하지만 효과적으로 task와 각 단일 노드와의 연관성을 정량화하였다. 
학습 과정 중 각 task의 representation은 서로 compositionality를 갖게 되며, 
이를 통해 특정 task를 타 tasks의 특성과 결합하여 수행할 수 있게 된다. 이는 인지의 유연성에 중요한 기본 원리로 보인다. 
또한 다중 작업 과제를 학습한 동물의 prefrontal neuron recording을 통해 얻은 데이터가 
continual-learning을 이용해 얻은 모델 데이터와 유사함을 보였다. 
이 연구는 곧 다양한 인지 과제를 수행할 때의 neural representation을 연구하는 중요한 플랫폼으로써 활용될 수 있을 것이다.**
  
  
**0. Clustering and Compositional embedding**  
- 논문의 큰 줄기를 위 두 단어로 요약할 수 있다.
- ![image](https://user-images.githubusercontent.com/46081019/63837846-306a9400-c9b7-11e9-8e75-524e8e98f9f9.png)  
  - fig 1a에서 multi-task neural representation의 encoding에 대한 4가지 가설을 제시한다.
    - 결론적으로 1사분면의 high clustering, high compositionality를 답으로 제안한다.
  - 굉장히 자주 등장하는 compositionality는 곧 task embedding 간의 독립성으로 생각하면 될 것 같다.
  - Composionality가 높다면 여러 task의 embedding이 entangling되어 복합적으로 performance에 영향을 줄 것이다.
  - 낮다면 각 representation은 2사분면처럼 독립적인 성격을 띄고 overlapping이 일어나지 않을 것이다.
  - 직관적으로 우리는 1사분면이 맞을 것이라 예상할 수 있는데, 우리는 특정 task를 배울 때 각 task의 sub-goal, sub-option 등을 분류하고 
  이를 다른 비슷한 task에 응용할 수 있기 때문이다.
  - 이번 논문의 contribution은 이러한 가설의 제기도 있지만 실험적으로 어떻게 이를 잘 보여주느냐에 있다.
- fig 1b는 실험 구조이다.
  - Input이 3가지로 나뉘는데, motor output의 시기를 결정하는 fixation cue, 간단한 stimuli cue(two modalities), rule inputs이다.
  - Rule inputs은 이 task가 몇 번째 task인지 알려준다. (1, 20)
  - 당연히 RNN은 1c처럼 20가지 task를 모두 잘 푼다고 가정하에 실험한다.
- ![image](https://user-images.githubusercontent.com/46081019/63839496-60fffd00-c9ba-11e9-8100-b72259a8b6bf.png)  
  - **실험 결과를 요약하면 recurrent hidden node가 functional하게 clustering된다는 것이다. **
  - fig 2a,b는 특정 hidden node output의 예시인데, 여러 task input에 대한 activation의 variance를 측정했더니 
  특정 task에서 특별히 유의미한 variance을 보였다는 것이다.
  - fig 2c,d는 구체적으로 tSNE를 했을 때 neuron이 군집화되는 점, 그리고 각 군집이 공통적으로 특정 class of task에 많이 반응한다는 점을 보인다. 
  - 예를 들어 5, 6번 클러스터는 DM, 8번은 working memory task 등에 특히 반응한다. 
  - 이는 전체 task에 대해 평균적으로 그렇다는 것이고, 각각의 training epoch을 보면 epoch마다 어떤 cluster가 주도적인지 다 다르다고 한다.
  - fig 2e는 각 cluster를 껐을 때 (neuro로 비유하면 'lesion'을 줬을때), cluster별로 특정 그룹의 task 성능이 확 감소하는 것을 볼 수 있다.
- ![image](https://user-images.githubusercontent.com/46081019/63840108-6f9ae400-c9bb-11e9-9e43-17954285c639.png)  
  - 이 clustering 및 compositional 패턴은 모델의 특성에 따라 조금씩 다르며, 특히 activation function의 영향을 많이 받는다고 한다.
  - activation function, RNN/GRU, intialization, L1 normalization을 바꿔가며 실험을 했을 때 클러스터의 개수가 제각각 다르다.
  - 이때 fig 3d,e,f,g를 보면 activation function 외에는 cluster 분포에 큰 영향이 없다.
  - 아마 activation function이 hyperplane geometry를 직접적으로 바꾸기 때문에 선택에 민감한 것으로 보인다.
  
**1. Metric of relationships between neural representation of pairs of task**  
- 각 hidden node가 특정 그룹의 task에 민감함을 보였다. 
  - 그러나 논문에서 주장하는 compositionality를 보기 위해서는 task마다 node를 따로 보지 않고 task 및 embedding 간의 유사성, 독립성 등을 보아야 한다. 
  - 여기서 제안하는 metric은 fractional task variance(FTV)로, 다음과 같다.
  - $$FTV_i(A, B) = \frac{TV_i(A) - TV_i(B)}{TV_i(A) + TV_i(B)}$$
    - TV는 앞서 사용한 variance of i-th neuron, A, B는 pair of task
    - 즉 모든 hidden node에 대해 FTV의 분포를 그려보면 hidden node가 A, B task에 대해서 얼마나 responsibility를 갖는지 볼 수 있다.
    - 만약 FTV가 -1 혹은 1에 굉장히 몰려있다면 hidden node가 독립적으로 하나의 task만을 인코딩한다고 할 수 있다.
  - ![image](https://user-images.githubusercontent.com/46081019/63840711-80982500-c9bc-11e9-9110-09be7b92b904.png)  
    - task가 20가지로, 총 190개의 FTV 조합이 나온다.
    - 각 조합에 대한 패턴을 크게 5가지로 요약할 수 있다고 한다.
    - 두 task의 embedding이 아예 포함 관계일수도, 어느 정도 겹칠 수도, 독립적이지만 부분적으로는 비슷할 수도 있다.
    - 가령 강아지 종을 분류하는 task와 고양이 종을 분류하는 task는 disjoint-mixed 혹은 equal이 나오지 않을까 싶다.
      - 강아지, 고양이 모두 공통적으로 색, 덩치등을 본다면 고양이는 수염, 강아지는 울음 소리를 보는 식으로?
      - 이를 tri-modal 패턴이라 한다.
- ![image](https://user-images.githubusercontent.com/46081019/63841039-074d0200-c9bd-11e9-99e2-2680a4022480.png)  
  - Trimodal 패턴을 갖는 pair of task을 예시로 들어 neural mechanism을 파악한 figure이다.
  - fig 5b를 보면 특정 task를 embedding하는 group(초록, 분홍)을 inactive하면 각 task의 performance에만 영향이 간다.
  - 그러나 overlapped lesions은 두 task 모두 inhibit한다.
  - fig 5d를 보면 이 겹치는 부분(하늘색)이 output과의 weight norm이 가장 높다고 한다.
  - 또한 fig 5e,f를 보면 서로 다른 두 group 1, 2(초록, 분홍)은 서로의 activation을 inhibition한다.
  - **개인적으로 이는 RNN이 lateral inhibition과 연관있음을 보여주는 것이 아닌가 생각한다.**
- ![image](https://user-images.githubusercontent.com/46081019/63841367-a5d96300-c9bd-11e9-8f3a-cd1e2639201b.png)  
  - 개인적으로 논문에서 가장 납득이 가지 않는 분석 방식이다.
  - 동일한 task에 대해 D-dimensional activation vector에 PCA을 돌린다.
  - 이를 통해 2d에서 서로 다른 두 task의 상관 관계를 분석한다.
    - 가령 fig 6b,c를 보면 (Anti, Dly Anti)와 (Go, Dly Go) vector가 서로 유사하기 때문에, 
    이 vector가 'Dly(delayed)' task 특성과 연관이 있고 이는 working memory cognitive process와 연관이 있다고 한다.
    - 하지만 모델이 RNN임을 감안하면 서로 다른 두 task가 delay된 것은 사실상 같은 task라고 보아야 하는게 아닌가 생각한다.
    - 다른 task 조합에 대한 비슷한 실험이 그 다음에 나오는데, 이는 보다 신뢰가 간다.
- ![image](https://user-images.githubusercontent.com/46081019/63841665-2c8e4000-c9be-11e9-8fce-fac9d453f2be.png)  
  - 개인적으로 가장 흥미로운 결과이다.
  - fig 1에서 정의한 rule input(혹은 task index)는 사실 강력한 input이기 때문에, 단순히 task의 indexing에 대해 activation이 augment된 것이 아니라 
  task 간의 composition을 학습할 수 있다는 것을 보였다.
  - fig 7a처럼 각 task에 대한 rule input을 개별적으로 학습시키고, test때는 rule input간의 relationship을 주었다.
    - 가령 7a에서는 (2번째 task) - (3번째 task) + (4번째 task)
  - 결과적으로 7b처럼 $$Dly Anti \sim Anti + Dly Go - Go$$으로 성능이 나왔다고 한다.
    - 7c는 $$Ctx Dly DM 1 \sim Ctx Dly DM 2 + Ctx DM 1 - Ctx DM 1$$
    - **이는 task를 각각의 'principal element'으로 학습했다고 할 수 있다.**
  - 7e에서는 모델을 freeze하고 이 rule input을 배우는 것만으로 transfer 관점에서 높은 성능을 보일 수 있다고 한다.
  
**1. Continual learning**  
- ![image](https://user-images.githubusercontent.com/46081019/63842130-0321e400-c9bf-11e9-930e-9b02492959f3.png)  
  - Continual learning을 통해 학습하였더니 더 잘 된다는 내용이다.
  - 동물은 catastrophic forgetting 문제에 강한데, 
  이는 "Decide to balance the need of learning with the need of retaining past memories" 능력이 있기 때문이라 생각한다.
    - 현재의 optimal 성능 혹은 탐색을 살짝 포기하더라도, 기존의 학습한 규칙과 task를 살짝 응용해서 효율적으로 문제를 풀어내는 것
  - 이번 논문의 단점 중 하나가 animal study와의 parallel connection이 부족하다는 것인데, 실제 PFC 데이터를 활용한 fig 8e가 이를 다소 보완한다.
    - Continual learning을 적용했더니 8d처럼 FTV 분포가 unimodal (특정 task에 처지지 않은)하게 바뀌었는데, 이것이 실제 분포와 비슷하다는 것
    - 즉 인간 역시 continual learning에 가깝게 학습하지 않을까 제안하고 있다.
  
**2. Conclusion**  
- PFC 혹은 lPFC가 이러한 task specialization에 관여한다는 점은 많이 알려져 있다.
  - 하지만 최근에는 brain의 기능을 specialization, modularization보다는 보다 complex하게 해석하려는 경향이 더 우세하다.
  - Information Coding의 효율성 측면에서 이는 맞는 수순이라 보이며, 특히 여러 neural population and representation의 overlapping이 
  뇌의 굉장한 효율성을 설명할 수 있는 key가 되지 않을까 싶다.
- Specialization과 Composition, 서로 충돌할 수 있는 두 가지 서로 다른 관점을 적절히 통합한 좋은 모델링 논문인 것 같다. 
  
