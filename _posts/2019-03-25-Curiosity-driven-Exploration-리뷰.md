---
title : "Curiosity-driven Exploration 리뷰"
use_math : true
layout : single
classes : wide
---
**0. 논문 소개**  
Reinforcement learning의 주요 과제는 크게 2가지로 나뉜다.  

1) Exploration-Exploitation dilema  
- 에이전트가 현재까지 학습한 액션(어쩌면 sub-optimal policy에 따른)을 고수할지, 추가적인 '정보'를 위해 다른 선택지를 탐험할지에 대한 문제. 
- 이때 탐험의 목표는 보상 그 자체일 수도, 혹은 미래의 가능한 보상을 위한 추가 정보 그 자체일 수 있음.  
2) Generalization  
- 현재 도메인을 넘어 타 도메인에 학습한 정책을 활용, 보다 효율적인 방식으로 적응.  
- 가장 해결이 어렵다고 여겨지는 문제  
    
이 두 가지 문제를 모두 해결한 획기적이었던 논문[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)을 정리하고자 한다. 
Curiosity-driven exploration 방식은 주어진 정보 중 agent-related feature을 걸러낸다.
그리고 취한 액션이 feature에 미치는 영향을 지속적으로 예측, 개선한다.
현재 state에서 특정 액션을 취했을 때 어떤 state'에 도달할 지를 끊임 없이 예측하고 있는 것이다.  
논문은 이 'prediction error'를 curiosity로 정의한다. 즉 에이전트의 행동 동기인 intrinsic reward을 만들어낸 것이다. 
  
인간의 뇌 역시 prediction machine 그 자체라고 볼 수 있다. 
보상 및 가치에 대한 정보를 인코딩하고 있는 Striatum의 dopamineric neuron은 예상과 다른 보상이 들어왔을 때 활발한 신경 활동을 보인다. 
그리고 이 prediciton 과정은 크게 reward와 관련이 없더라도 지속적으로 이뤄지고 있다. 
가령 Primary sensory neuron population은 주어진 감각 자극에 대한 population coding을 통해 끊임 없이 자극의 정확한 위치를 예측하고 개선한다. 
([뉴런의 Bayesian Inference](https://www.ncbi.nlm.nih.gov/pubmed/17057707))  
보상과 크게 관련이 없지만 지속적으로 환경을 탐색하는 행동은 prior knowledge가 부족한 영유아에게서도 흔히 관찰되는 현상이다. 
즉 인간은 어떤 행동에 대한 외부적 보상 외에도 고유한 동기를 갖고 있다는 뜻이 되겠다. 
이러한 점에서 이번에 소개할 논문은 상당히 **Neural-friendly method**라고도 할 수 있다.  
  
앞서 설명한 인간의 행동 패턴과 현재 강화 학습 알고리즘의 패러다임은 큰 차이를 보이고 있다. 기존 패러다임은 extrinsic reward을 휴리스틱하게 정의해야 한다는 점이다. 
매우 복잡한 task, 혹은 다양한 task에서는 이 reward의 합리적인 정의 자체가 쉽지 않다. 
따라서 굉장히 확실한 행동에만 sparse하게 reward를 assign하는 경우가 많다. 이는 sparse-reward 문제로 불리며 exploration을 방해하는 요소이다. 
최근에는 GAN 등의 생성 모델을 통해 reward를 보다 dense하게 유지하거나, hierarchical RL을 통해 다양한 sub-goal을 정의하려는 움직임이 각광받고 있다. 
  
**1. 모델 구조**  
![image](https://user-images.githubusercontent.com/46081019/54964809-30eaa300-4fb1-11e9-98c5-ad3fe3001152.png)  
에이전트는 액션 생성 부분과 intrinsic reward 생성 부분으로 나뉜다. 액션 baseline은 A3C를 사용했고, 
advantage 계산을 위한 reward 정의 과정에서 extrinsic reward($$r_t^e$$)와 intrinsic reward($$r_t^i$$)가 더해진다. 
이 Intrinsic reward이 곧 curiosity를 의미하며 논문의 핵심인 Intrinsic Curiosity Module(ICM)에서 계산된다. 
내부 scaling을 거친 intrinsic reward가 extrinsic reward와 1대1로 더해진다.
**1.1 ICM**  
ICM은 다시 Inverse model과 Forward model로 나뉜다. 
- Inverse model : agent와 관련 있는 state feature 학습
- Forward model : state feature의 예측 오류(prediction error)
을 목표로 한다.   

논문의 주요 novelty는 첫 번째 모델인 Inverse model에서 나오는데, 
기존 inverse-forward model을 사용한 논문은 raw pixel-space을 그대로 사용하여 
agent가 실제로 액션 및 상태 변화에 중요하지 않은 환경 요소(배경 등)에 distract되는 문제가 있었다. 
반면 Inverse model은 state feature $$\phi(s_t)$$와 $$\phi(s_{t+1})$$을 입력으로 받아 $$a_t$$를 학습하기 때문에 
정확히 state의 어떤 요소가 액션에 의해 변화했는지 알 수 있다. 
즉 inverse model에서 파생된 loss 및 gradient의 back-propagation을 통해 state의 essence인 $$\phi(s_t)$$을 올바르게 학습할 수 있다.  
  
실제 실행한 action $$a_t$$와 $$\phi(s_t)$$를 concatenate하여 forwarding하면 $$\phi(s_{t+1})$$을 예측할 수 있다. 
$$\phi(s_{t+1})$$와 예측한 값의 오차를 intrinsic reward로 정의한다. 즉 forward model에 기존에 겪어보지 못한 $$\phi(s_{t+1}), a_t$$을 넣으면 높은 intrinsic reward을 얻을 수 있다. 익숙하지 않은 (s,a)의 탐험을 incentivize하는 것이다.   
기존 extrinsic reward에 의존할 때는 reward가 굉장히 sparse하지만, intrinsic reward을 정의함으로써 reward space가 보다 dense해지며, 
state-space에 대한 novelty seeking이 가능해진다.  
  
최종 loss는 policy loss, inverse model loss, forward model loss을 weighting하여 정의된다.  
  
**2. Results**  
개인적으로 아이디어 외에도 이를 검증하는 방법론 자체가 상당히 인상적이었다. 
두 가지 결과가 핵심이라 생각하는데, Input에 noise를 넣어 robust한 결과를 얻은 것과 extrinsic reward을 아예 배제하고 결과를 얻은 것이다.
- Input noise  
![image](https://user-images.githubusercontent.com/46081019/54968364-c2f8a880-4fbd-11e9-8125-47194843fb77.png)   
알고리즘은 위의 input+noise에 대해 robust한 결과를 얻었는데 이는 기존 raw-pixel 기반의 접근 방식과 현 알고리즘의 차이를 대표적으로 보여주고 있다. 
주어진 state-space에서 실제로 중요한 부분이 어디인지 제대로 학습했다는 증거이다. (agent action에 의한 직접적인 변화+agent가 컨트롤할 수 없지만 영향을 미치는 변화) 실제 자연 지능의 행동 양식과도 비슷한 알고리즘이라는 점이 느껴진다. 일종의 선택적 호기심이 작용한 결과라고 할 수 있겠다.  
  
- Intrinsic reward only  
![image](https://user-images.githubusercontent.com/46081019/54968662-d0faf900-4fbe-11e9-977a-86d52107347e.png)  
VizDoom의 한 맵에 에이전트를 트레이닝시키고 이를 타 맵에 적용시킨 generalization 실험 결과이다.
물론 Extrinsic reward와 함께 fine-tuning했을때 가장 성능이 좋지만 Intrinsic reward만을 도입했는데도 불구하고 (Yellow) 괜찮은 generalization 성능을 보인다. 이 때 마찬가지로 raw-pixel을 통해 실험한 경우 좋지 않은 성능을 보인다.  
흥미롭게도 마리오 게임에서 world 1에 intrinsic reward을 학습시킨 다음 world 3에서 generalization 실험을 거쳤을 때 제대로 작동하지 않았다고 한다.논문에서는 이를 level 3의 난이도가 지나치게 높아 curiosity reward만으로는 학습이 어려운 장벽에 부딪혔다고 설명한다. 
새로운 state를 계속해서 무작위하게 탐색하기보다는 외부 reward에 대한 information이 어떤 형식으로든 제공되야 한다는 것을 의미한다.
(에이전트의 interaction, 휴리스틱한 extrinsic reward 정의 등)  
  

