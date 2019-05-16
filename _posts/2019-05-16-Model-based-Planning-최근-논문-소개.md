---
title: "Model based Planning 최근 논문 소개"
use_math: true
classes: wide
layout: single
---

*다음을 주로 참고하였습니다*  
- [I2A](https://arxiv.org/abs/1707.06203)
- [Learning and Querying Fast Generative Models for RL](https://arxiv.org/abs/1802.03006)
- [Deepmind blog](https://deepmind.com/blog/agents-imagine-and-plan/)
- [Learning improved dynamics model in RL by incorporating the long term future](https://arxiv.org/abs/1903.01599)
- [Learning model-based planning from scratch](https://arxiv.org/abs/1707.06170)  

강화 학습의 SOTA는 대부분 value-based or policy-based의 model-free 알고리즘이 차지하고 있다. 
이는 환경 고유의 underlying dynamics을 explicit하게 고려하지 않은 것이며, 따라서 많은 training 데이터가 요구된다. 
하지만 인간 및 자연 지능은 MF에만 의존하지 않고, MB 시스템 역시 가지고 있는 것으로 보여진다. 
유명한 연구로는 [N.Daw paper](http://www.princeton.edu/~ndaw/dnd05.pdf)에서 다룬 reward uncertainty와 MB-MF model arbitration이 있다. 
우리 주변의 환경은 지속적으로 변화하기 때문에 단순히 MF에만 의존한다면 쉽게 원하는 것을 얻기 어려울 것이다. Underlying Transition Mechanism을 
더 잘 이해하는 species가 생존 경쟁에 유리한 것이다. 
반대로 Model에 대한 이해와 지속적인 prediction, motor action이 필요 없는 종(e.g., 바위에 정착하는 해양생물)은 
뇌가 아예 없거나 필요에 따라 스스로 제거하기도 한다.
  
앞으로 어떻게 효과적으로 model을 구축하고, planning을 구현할 것인지에 대해서 더 활발하게 연구가 진행될 것이라 생각한다. 
딥마인드의 수장인 Demis Hassabis는 예전부터 Hippocampus와 Imagination, Episodic Memory에 대한 연구를 해왔는데, 
이 역시 AI가 어떻게 인간의 계획, 상상 능력을 배울 수 있을지에 대해 고민한 흔적이라고 보인다.
- [Patients with hippocampal amnesia cannot imagine new experiences](https://www.pnas.org/content/104/5/1726)
- [Using Imagination to Understand the Neural Basis of Episodic Memory](http://www.jneurosci.org/content/27/52/14365)  
나 역시 자연 지능이 어떻게 모델을 기억하고 이를 행동 계획, 상상에 활용하는지를 이해하는 것이 궁극적인 AGI 개발에 있어 중요하다고 생각한다. 

딥마인드 외에도 다양한 그룹 (벤지오 교수님, 이홍락 교수님 등)이 planning에 많은 관심을 갖고 연구를 진행 중이신 것으로 보인다.
여기서는 pixel-based planning의 성공적인 연구로서 많은 관심을 받은 I2A을 시작으로 
몇몇 연구들이 model을 어떻게 인코딩하고 있는지, 이를 어떻게 planning에 활용하고 있는지 정리한다. 
  
  
**1. Imagination-Augmented Agents for Deep Reinforcement Learning**
- NIPS 2017, Deepmind
![image](https://user-images.githubusercontent.com/46081019/57821375-45863300-77cb-11e9-954d-30671d7fd6c2.png)  
오른쪽 figure처럼 I2A는 현재 상황에서부터 다양한 시나리오를 roll-out할 수 있는 알고리즘이다. 
Model-based 모듈과 Model-free 모듈이 공존한다는 점이 타 model-based research와 구분되는 차이점이다.  
![image](https://user-images.githubusercontent.com/46081019/57821660-42d80d80-77cc-11e9-940f-407cf5535b4a.png)  
전체 모델은 크게 Imagination core, Roll-out, Aggregation 파트로 나뉜다. 
Imagination core에서 환경의 dynamics을 배우고, 이를 기반으로 roll-out에서 가능한 trajectory를 상상하여 이 정보를 Model-free path와 
합치는 것이다. 이 모델 학습, 상상으로 이어지는 Model-based path가 없으면 Model-free와 동일한 에이전트이다. 
Policy, value를 학습함에 있어 MF의 관측 정보에만 의존하지 않는 것이다. 
  
- Imagination Core  
![image](https://user-images.githubusercontent.com/46081019/57821867-1f619280-77cd-11e9-8afb-f47638e215bf.png)  
Model 학습 방식이 재밌는데, input action을 tiling하여 input observation의 long-term spatial 정보를 배웠다. 
$$O_t, a_t$$를 받아 $$o_{t+1}, r_{t+1}$$를 리턴하는데, 이는 model-transition을 deterministic하게 가정한 것이다. 
Imagination Core에서 internal action을 결정할 때는 internal policy을 따르는데, 
이는 최종적으로 모든 정보를 갖고 학습된 aggregated policy와 비슷하도록 만든다. 
이를 위해 논문에서는 두 policy의 cross entropy를 전체 학습의 auxiliary loss로 활용한다. 
이를 통해 에이전트가 상상한 trajectory가 별 의미, 가치 없는 trajectory가 아닌 보다 중요한 trajectory라고 assure할 수 있다는 게 주장이다.
사실 이때 더 중요한 과정은 바로 다음에 나오는 encoding이라고 생각한다. 
  
- Rollout and encoding  
Imagination core의 internal policy 자체는 완벽하지 않을 수 밖에 없다. 
Non-sensical prediction이 포함되어 있을 수 있고, 
지금 당장 reward와는 큰 연관이 없더라도 trajectory에 또 다른 중요한 정보가 포함되어 있을 수도 있다. (e.g., 게임의 중요한 rule를 배울 수 있는 정보)  
이에 저자는 n-step backward LSTM 모델을 trajectory의 encoder로 사용한다. 
이 과정을 Learn-to-interpret이라고 부른다. 
전체 학습 과정에서 LSTM은 valuation과 관련된 중요 transition 정보를 기억하게 되고, 
또 다른 trajectory를 확인했을 때 value와 관련 높은 정보를 추출하거나, 필요 없는 trajectory인 경우 무시(forget)하는 것이 가능하다. 

- Aggregator  
LSTM의 output을 concatenate하여 'single imagination code'을 얻는다. 동일한 output을 통해 얻은 
