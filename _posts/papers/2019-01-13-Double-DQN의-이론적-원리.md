---
title: "Double DQN의 이론적 원리"
use_math: True
layout: single
classes: wide
---

**본 포스트는 다음 논문들을 기반으로 작성되었습니다. 참고한 정도로 정렬했습니다.**  
- "Double Q-learning", *Hado van Hasselt*, 2010
- "Deep Reinforcement Learning with Double Q-Learning", *David Silver*, 2015
- "Estimating the Maximum Expected Value: An Analysis of (Nested) Cross Validation and the 
Maximum Sample Average", *Hado van Hasselt*, 2016  
  
  
**0. Double DQN(DDQN)이란?**  
DDQN = DQN + Double Q-learning이라 생각합니다. Q-learning, DQN(Deep Q Network)에 관한 설명은 많은 자료가 있으므로 생략하겠습니다.  
Double Q-Learning은 기존에 존재했던 Q-Learning 개선 알고리즘입니다. DQN의 등장 이후 두 알고리즘이 합쳐져 지금까지도 널리 쓰이고 있는 DDQN으로 발전했습니다.  

**1. Double Q-Learning이란?**  
Stochastic한 환경에서 Q-Learning은 종종 좋지 않은 성능을 보이며, 그 원인으로 Overestimation이 오래 전부터 거론되었습니다. 
또한 Overestimation의 원인으로는 Q-learning의 Maximum Estimation(ME)이 거론되었는데요. 
각종 포스트 및 DDQN paper에서 이 overestimation에 대한 맘에 드는 설명을 찾기 어려웠습니다.  
한편 reddit에서 저자가 직접 질문에 답한 discussion이 있는데요.  
<https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/>  
인용하면 다음과 같습니다.  
> Overestimation is due to the fact that the expectation of a maximum is greater than or equal to the maximum of an expectation, 
often with a strict inequality (see, e.g., Theorem 1 in https://arxiv.org/abs/1302.7175). So, yes, there is a theoretic motivation behind the algorithm.    

위 답변을 기반으로 overestimation에 대해 알아보고자 합니다.
 
**2. Overestimation의 원인**  
우선 Q-Learning의 목표 object는 **maximum of expected value of the next state s'**, $$max_{a'}(E(Q(s', a'))$$입니다. 
(s,a)를 실행할 때, 얻어지는 보상 + s'에서 기대되는 maximum action value가 곧 (s,a)의 가치를 대변하게 됩니다. 
즉 s'에서 취할 수 있는 actions 중 어느 action의 value 기댓값이 가장 클 것으로 예상되는지가 궁금한 것이죠. (정확히는 off-policy기 때문에 a'보다는 Q(s',a')가 관심입니다.) 
이때 Q-Learning에서는 **위의 목표를 estimate하기 위해** $$max_{a'}(Q(s', a'))$$을 사용합니다.  
하지만 이때 이 estimation에는 non-negative bias가 존재합니다. 즉, $$E(max_{a'}(Q(s', a'))) \geq max_{a'}(E(Q(s', a')) $$입니다. 
$$max_{a'}(Q(s', a'))$$는 *평균적으로* 어떠한 a' action-value의 기대값보다도 크다는 뜻입니다. 
즉 a'의 action-value을 실제보다 더 높은 값으로 판단해버리는, over-estimation이 발생합니다.  

논문에서는 이를 single estimator의 문제점으로 규정하고, 이에 대한 대안으로 double estimator를 제안합니다.  

**2.1 single estimator 보충 설명**  
s'에서 취할 수 있는 3개의 action $$a_1', a_2', a_3'$$이 있고, 각 action value $$Q(s', a_i')$$를 random variable X_i라 합시다.  
이때 $$max_iE(X_i)$$를 estimate해야 합니다. 이를 위해 각 action value의 experience sample 집합 $$S_i$$를 생각합시다. (experience sample은 편의상 제가 명명했습니다; 강화학습의 experience replay에서 따온 느낌으로 보시면 될 것 같습니다)   

각 sample의 평균 중 최대값 $$max_i \mu_i(S)$$을 찾으면 $$max_iE(X_i)$$을 estimate하는 것이라는 주장이 Q-learning이자 single-estimator 방식입니다.   
![image](https://user-images.githubusercontent.com/46081019/51086360-70169f80-1789-11e9-97c9-d6c98d4348ed.png)   
하지만 $$max_i \mu_i(S)$$은 **$$E(max_i \mu_i)$$의 unbias sampled estimate**이며, 이는 앞서 말했듯이 **maximum of expectation, 즉 $$max_iE(X_i)$$보다도 큰 값을 갖습니다.**  

왜 이러한 크기 관계가 성립하는지 Informal하게 이해해보자면, $$E(max_i \mu_i)$$의 경우 alongside with $$S_i$$를 따라 그때그때 $$max_i \mu_i$$를 구하는 반면, $$max_iE(X_i)$$의 경우 전체적인 $$S_i$$를 보고 expected value가 가장 큰 i(=action-value 기대값이 가장 큰 액션)을 고른다고 할 수 있겠습니다. 그림으로 표현하면 다음과 같습니다.
![1](https://user-images.githubusercontent.com/46081019/51093720-d6330f00-17e9-11e9-92ad-44918bb7c756.png)  

Formal한 증명은 past paper에 있지만 확인은 하지 못한 상태입니다.  

**2.2 double estimator**  
Double-DQN에서 single estimator를 대체하여 사용하는 방식은 double-estimator입니다. 논문에서는 e-greedy하게 action을 select하는 selection 과정과, 이러한 e-greedy policy 자체를 평가하는 evaluation 과정을 분리했는데요. 이를 'double' estimator의 예시라고 할 수 있습니다. 요지는 다음 식과 같습니다.  
![image](https://user-images.githubusercontent.com/46081019/51093805-8e60b780-17ea-11e9-8da2-fba96e043a03.png)  
여기서 A, B는 각각 experience sample 집합 S를 independent하게 split한 subset $$S^A, S^B$$입니다. 새로 등장한 인덱스 $$a^{*}$$는, **$$max_i \mu_i^A(S^A)$$인 i의 집합입니다.**   
이 방식은 Double-DQN 알고리즘과 동일한데요. Double-DQN에서 main network가 maximum으로 뽑은 'action'의 집합으로 target network를 indexing하여 최종 objective action-value를 얻게 됩니다. 위의 알고리즘 역시 $$a^{*}$$로 $$\mu^B$$를 indexing하고 있습니다. 아래 그림을 참고해 주세요.  
![1](https://user-images.githubusercontent.com/46081019/51096395-8f501400-17ff-11e9-9bd5-d31d725c7bb0.png)  

그렇다면 이 indexing이 (maximum) expectation of action value를 unbias하게 estimate하고 있을까요? Double-estimator는 single estimator와는 반대로,  overestimate가 아닌 underestimate를 하게 됩니다. 아래 Lemma 1이 이를 증명하고 있습니다.  
![image](https://user-images.githubusercontent.com/46081019/51096417-ca524780-17ff-11e9-80a8-206467dd60c9.png)  

**3. conclusion**  
Double-DQN은 stochastic한 상황에서 maximum estimator가 action value를 제대로 estimate하지 못하는 점을 개선하기 위한 알고리즘입니다. 논문에도 잘 나와 있듯이 다양한 시뮬레이션 환경(real-action value function approximator, Atari games)에서 잘 작동되는 것을 확인하였고, 지금까지도 많이 활용되는 알고리즘입니다. 또한 이 포스트에서는 리뷰하지 않았지만 Double Q-learning을 극한으로 보냈을 때 optimal policy로 수렴하게 된다고 합니다.
