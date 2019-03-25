---
title : "Curiosity-driven Exploration 리뷰"
use_math : true
layout : single
classes : wide
---
**0. 논문 소개**  
Reinforcement learning의 주요 과제는 크게 2가지로 나뉩니다.  

1) Exploration-Exploitation dilema  
- 에이전트가 현재까지 학습한 액션(어쩌면 sub-optimal policy에 따른)을 고수할지, 추가적인 '정보'를 위해 다른 선택지를 탐험할지에 대한 문제. 
- 이때 탐험의 목표는 보상 그 자체일 수도, 혹은 미래의 가능한 보상을 위한 추가 정보 그 자체일 수 있음.  
2) Generalization  
- 현재 도메인을 넘어 타 도메인에 학습한 정책을 활용, 보다 효율적인 방식으로 적응.  
- 가장 해결이 어렵다고 여겨지는 문제  
    
이 두 가지 문제를 모두 해결한 획기적이었던 논문[Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)을 정리하고자 합니다. 
Curiosity-driven exploration 방식은 주어진 정보 중 agent-related feature을 걸러냅니다.
그리고 취한 액션이 feature에 미치는 영향을 지속적으로 예측, 개선합니다.
현재 state에서 특정 액션을 취했을 때 어떤 state'에 도달할 지를 끊임 없이 예측하고 있는 것이지요.  
논문은 이 'prediction error'를 curiosity로 정의합니다. 즉 에이전트의 행동 동기인 intrinsic reward을 만들어낸 것입니다. 
  
인간의 뇌 역시 prediction machine 그 자체라고 볼 수 있습니다. 
보상 및 가치에 대한 정보를 인코딩하고 있는 Striatum의 dopamineric neuron은 예상과 다른 보상이 들어왔을 때 활발한 신경 활동을 보입니다. 
그리고 이 prediciton 과정은 크게 reward와 관련이 없더라도 지속적으로 이뤄지고 있습니다. 
가령 Primary sensory neuron population은 주어진 감각 자극에 대한 population coding을 통해 끊임 없이 자극의 정확한 위치를 예측하고 개선합니다. 
([뉴런의 Bayesian Inference](https://www.ncbi.nlm.nih.gov/pubmed/17057707))  
보상과 크게 관련이 없지만 지속적으로 환경을 탐색하는 행동은 prior knowledge가 부족한 영유아에게서도 흔히 관찰되는 현상입니다. 
즉 인간은 어떤 행동에 대한 외부적 보상 외에도 고유한 동기를 갖고 있다는 뜻이 되겠습니다. 
이러한 점에서 이번에 소개할 논문은 상당히 **Neural-friendly method**라고도 할 수 있겠습니다.  
  
앞서 인간의 행동 패턴과 강화 학습 알고리즘의 패러다임은 큰 차이를 보이고 있습니다. 기존 패러다임은 extrinsic reward을 휴리스틱하게 정의해야 한다는 점입니다. 
매우 복잡한 task, 혹은 다양한 task에서는 이 reward의 합리적인 정의 자체가 쉽지 않습니다. 
따라서 굉장히 확실한 행동에만 sparse하게 reward를 assign하는 경우가 많습니다. 이는 sparse-reward 문제로 불리며 exploration을 방해하는 요소입니다. 
최근에는 GAN 등의 생성 모델을 통해 reward를 보다 dense하게 유지하거나, hierarchical RL을 통해 다양한 sub-goal을 정의하려는 움직임이 각광받고 있습니다. 
  
**1. 
