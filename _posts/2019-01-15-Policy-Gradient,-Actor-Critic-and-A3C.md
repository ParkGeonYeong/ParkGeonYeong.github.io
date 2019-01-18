---
title: "Policy Gradient, Actor Critic and A3C"
use_math: True
layout: single
---

본 글은 다음 자료를 주로 참고하여 작성하였습니다.  
- Policy Gradient Lecture, *David Silver*
- "Asynchronous Methods for Deep Reinforcement Learning", *Mnih. V., et al*
- Medium blog posts of *Arthur Juliani*  

**0. Policy Gradient란 무엇인가?**  
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
- 여기서 loss는 $$log{\pi}_{\theta}(s,a)r$$이며, parameter $$\theta$$로 gradient를 구하고 있습니다.
- 다만 이 'loss'는 최소화가 아닌 최대화의 대상이기 때문에, gradient ascent로 접근해야 합니다.
- 이를 gradient descent 문제로 바꾸기 위해 loss에 -를 곱해줍니다.  
- **loss = $$-log{\pi}_{\theta}(s,a)A$$
- 이때 A는 'advantage'입니다.
  
**1. Actor-Critic이란 무엇인가?**  
앞서 policy gradient의 최종 loss를 정의함에 있어 instantaneous한 reward인 r을 사용했고, 보다 일반화된 표현으로 바꿔주기 위해 advantage term A를 사용했습니다. 사실 이 term A에는 instant reward r외에도 여러 값들이 들어갈 수 있는데, long-term $$Q^{\pi}$$가 그 예시입니다. 하지만 우리는 이 ground-truth $$Q^{\pi}$$를 모르기 때문에 (사실 이것을 알면 이미 문제는 풀린 것이기에), 이에 대한 대체로 unbiased sample r을 사용한 것입니다.  
  
여기서 우리는 자연스럽게, reward r 말고 우리가 기존에 해왔던 Q-learning, DQN에서처럼 **$$Q_w$$를 학습해서 advantage term A를 대체**할 수 있지 않을까라는 질문을 할 수 있습니다. 이것이 바로 **Critic**의 개념입니다. 즉 critic으로 action-value function을 estimate하고, 이 값을 사용하여 actor가 policy를 update하는 것입니다. 마치 Double-DQN처럼 double estimator를 활용하여 action value와 policy를 개별적으로 평가하고 있습니다. [Double DQN의 double estimator를 확인해 보세요](https://parkgeonyeong.github.io/Double-DQN%EC%9D%98-%EC%9D%B4%EB%A1%A0%EC%A0%81-%EC%9B%90%EB%A6%AC/)  
   
![image](https://user-images.githubusercontent.com/46081019/51327647-21347700-1ab5-11e9-8e16-8cb8d764f79f.png)   
위 슈도-알고리즘에서 확인할 수 있듯이 개별적인 action-value estimator $$Q_w$$를 업데이트해가며, 해당 $$Q_W(s,a)$$값을 policy gradient에도 사용하고 있습니다. 이때 monte-carlo 방식과 달리 현재는 online 학습이 가능합니다. (episode가 끝나기 전에 계속 학습중). 한 가지 Q-learning 등과 비교해서 주의해야 하는 차이점은, actor-critic 방식에서 action을 결정하는 원인이자 주체는 policy $$\theta$$이며, $$Q_w$$가 아니라는 점입니다. Action-value estimator는 말 그대로 estimator로서 어떤 힌트만 제공할 뿐, 현재 decision-making의 주체는 policy 그 자체입니다.  
  
이 action-value estimator는 사실 다양한 버전이 가능합니다.    
한 가지 방법은 바로 advantage function $$A^{\pi_{\theta}}(s,a)$$를 Q 대신 사용하는 것입니다. 이 advantage function은 간단히 설명하면 Q(s,a)에서 V(s)를 뺀 값입니다. 지금 어떤 액션을 취했을 때, 현재 머무르고 있는 state s의 가치 V 대비 얼마나 더 이득을 얻을 수 있을지를 다루고 있습니다. 이는 [dueling DQN](https://arxiv.org/abs/1511.06581)과도 유사한 개념이라 보입니다.  
![image](https://user-images.githubusercontent.com/46081019/51328659-39a59100-1ab7-11e9-8da2-d11223708cee.png)   
이때 $$A^{\pi_{\theta}}(s,a)$$를 대신 사용해도 괜찮은 이유는, action과 관련없는 state function(such as $$V(s)$$는 현재 gradient에 더하거나 빼도 기대값에 변화를 주지 못하기 때문입니다.  
  
또 다른 중요한 트릭으로는 바로 TD error의 활용입니다. 결론부터 말하면 advantage function은 사실 TD error로 estimate가능합니다. TD error는 특정 action을 취했을때 기대되는 보상 및 미래 가치 - 현재 state, action의 가치이기 때문입니다. **따라서 최종적으로 actor-critic의 gradient는  
$$
\triangledown_{\theta}J(\theta) = E_{\pi_{\theta}} [\triangledown_{\theta}log{\pi_{\theta}}(s,a)\delta^{\pi_{\theta}}]
$$  
의 형태로 많이 사용합니다.** (이때 delta가 TD error를 의미합니다)  앞에서 r을 Q로 대체한 것처럼, TD error도 결국은 approximate estimator이기 때문에 에이전트의 experience에 의해 variance 영향을 많이 받습니다. 따라서 이를 고려해 $$TD(\lambda)$$를 사용하기도 합니다.
