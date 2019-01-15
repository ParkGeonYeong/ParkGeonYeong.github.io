---
title: "Policy Gradient, Actor Critic and A3C"
use_math: True
---

본 글은 다음 자료를 주로 참고하여 작성하였습니다.  
- Policy Gradient Lecture, *David Silver*
- "Asynchronous Methods for Deep Reinforcement Learning", *Mnih. V., et al*
- Medium blog posts of *Arthur Juliani*  

**0. Policy Gradient란 무엇인가?*  
2016년 나와 아직까지 핫한 알고리즘으로 자리잡고 있는 A3C(Asynchronous advantage actor-critic) 알고리즘은 actor-critic을 이해해야 하고, 
actor-critic은 policy gradient를 이해해야 합니다.  
Policy Gradient는 action 혹은 state의 value를 approximate하여 문제를 해결하는 value-based RL 알고리즘과는 다르게 policy 자체를 학습의 object로 규정합니다. 
기존의 알고리즘은 이 policy가 학습의 대상이라기보단 implicit하게 규정되어 있었습니다; e-greedy, greedy, etc. 하지만 policy gradient는 이 policy를
 학습의 대상으로 여겨 (이를 parametrise한다고 표현합니다) 현재 state를 관찰한 후 각 possible action에 대해 weight를 부여합니다.  
 ![image](https://user-images.githubusercontent.com/46081019/51182246-e1cd2580-1910-11e9-992c-5028cc951f86.png)  
위의 잘 알려진 도표가 policy 접근법과 value 접근법의 차이를 잘 보여주고 있습니다.  
   
Policy gradient 접근법이 value-based 접근법 대비 갖는 장점은 몇가지가 있지만 대표적으로는 stochastic한 probability를 학습하는 것이 수월하다는 점입니다. 
Value-based 접근법은 value function을 학습 parameterise하기 때문에, 이에 대응되는 decision-making (policy)는 deterministic한 성향을 보입니다. 
가령 가위바위보를 할때 가위의 action value를 조금이라도 높게 설정한 에이전트가 있다면 그 에이전트는 implicit policy(e-greedy)에 의해 가위를 상당히 자주 선택할 것입니다. 
하지만 policy gradient의 경우 해당 action value의 작은 변화에 큰 영향을 받지 않습니다. 대신 각 action에 weight 형식으로 배정된 최적의 probability를 학습해서 
가위, 바위, 보를 거의 균등하게 내게 됩니다. (물론 현재 예시라면 가위를 확률적으로 조금 더 낼 수 있겠지만 이는 stochastic한 환경에서 당연한 일입니다)   

언급했듯이 policy gradient는 policy 자체를 parameterize합니다. 현재 state s에서 policy parameter를 $$\theta$$로 규정하겠습니다. 
이 상황에서 policy에 의해 action a를 취할 확률은 $$\pi_{\theta}(s,a) = P[a|s,\theta]$$으로 표현할 수 있습니다. 
**즉 policy gradient는 parameter $$\theta$$를 조절해 얻을 수 있는 가치를 최대화하는 optimization 문제로 치환할 수 있습니다.**  

Optimization 문제를 풀기 위해, 먼저 Objective function을 규정하고 이를 통해 loss를 정의해야 합니다. 
Objective function은 강화 학습의 목표인 'expectation of something valuable'을 최대화하는 것으로 생각할 수 있습니다. 
이를 위해 Time-step마다 들어오는 Reward, 혹은 average value, 혹은 장기적인 $$Q(s,a)$$ 등을 생각할 수 있습니다. 형식은 다음과 같습니다. 
이번 포스트에서는 맨 밑의 Reward Objective function을 사용합니다.
![image](https://user-images.githubusercontent.com/46081019/51182956-2ce83800-1913-11e9-997d-0f720435359b.png)  

유의할 점은 일반적인 머신러닝에서 흔하게 접하는 objective function과는 느낌이 다르게, 강화학습에서는 현재 $$J(\theta)$$를 최대화하도록 $$\theta$$를 
찾아야 하기 때문에, 파라미터 $$\theta$$에 대해 gradient descent가 아닌 gradient ascent로 접근해야 한다는 것입니다. 
(물론 현재 문제 세팅 상에서는 그렇고, 이를 gradient descent 문제로 바꿔 풀 수 있습니다) 
위 objective function에서 $$\pi_{\theta}(s,a)$$가 $$\theta-dependent$$합니다. $$\theta$$에 대해 gradient를 구해보면 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/46081019/51184149-c9600980-1916-11e9-8705-0ba9385881b0.png)   
나중에 나오겠지만, $$\pi$$를 곱하고 나누는 트릭을 통해 Gradient의 expectation을 구할 수 있습니다. $$\theta-dependent%%한 $$log\pi_{\theta}(s,a)$$를 score function으로 규정합니다. 
  
이 term은 parameter $$\theta$$를 통해 각 action의 policy-driven weight(probability)를 할당하고 있는데, 어떤 확률 함수를 사용하는가에 따라 policy의 형태가 달라집니다. 
대표적인 softmax policy는 다음과 같습니다. Parameter $$\theta$$와 현재의 (s,a) feature가 곱해진 형태입니다.
$$\pi_{\theta}(s,a) \propto e^{\phi(s,a)^{\intercal}\theta}$$   
  
이 softmax policy를 사용했을때 score function은 다음과 같이 유도할 수 있습니다.  
  
/* TO BE FILLED */  
  
이 score function을 black box로 취급하고, general한 policy 형태에 대해 gradient를 일반화하면 다음과 같습니다.  
![image](https://user-images.githubusercontent.com/46081019/51185128-a71bbb00-1919-11e9-84fb-99888610e3ee.png)  
- 위 식에서 score function 외의 $$\pi$$를 빼놨기 때문에 기대값 형태로 쓸 수 있습니다.
- 여기서 loss
