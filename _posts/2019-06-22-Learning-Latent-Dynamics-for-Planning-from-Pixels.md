---
title: "Learning Latent Dynamics for Planning from Pixels"
use_math: true
layout: single
classes: wide
---  
  
  
본 자료는 다음 논문을 정리했습니다.  
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  
  
  
다음 포스트와 다소 연관이 있습니다. 
- [Recent Works about MB and Planning](https://parkgeonyeong.github.io/Model-based-Planning-and-some-recent-works/)  
- [Deep Variational Bayes Filter](https://parkgeonyeong.github.io/Deep-Variational-Bayes-Filters;-Non-linear-State-Space-Model-%ED%95%99%EC%8A%B5/)  
  
  
**요약**
이번 ICML 2019에 oral로 발표된 Google Brain의 논문이며, 이홍락 교수님이 포함되어 계신다. PlaNet(planning network)라고도 불린다.  
Latent dynamics을 배우고, 이를 기반으로 Fast Online planning in latent space를 목표로 한다는 점에서 
기존 state-space planning 연구들과 비슷하다. 그러나 2가지 정도의 major novel contribution(RSSM, Latent Overshooting)이 보인다. 
무엇보다 어려운 도메인(Continuous motor control)에 대해서도 sequential VAE가 잘 prediction을 하고 있는 점이 인상적이다. 


  
**0. Algorithms**  
![image](https://user-images.githubusercontent.com/46081019/59933102-b0d6bb00-9483-11e9-950f-1c89eb85adfc.png)  
- 알고리즘의 골격 
- Model = Transition Model($$p(s_t \mid s_{t-1}, a_{t-1}$$) + Observation decoder($$p(o_t \mid s_t)$$) + Reward Model($$p(r_t \mid s_t)$$) 
- Observation Decoder는 $$s_t$$의 feature space를 제한할 뿐 planning에 직접 활용하지는 않는다.
  - Observation space가 아닌 Latent space를 기반으로 빠르게 planning하는게 핵심이다. 
- Experience = Initially Collected by partially trained model
  - 처음에 random action set을 모델에 학습시킨다. 이 데이터셋은 C update steps마다 확장한다.
  - 또한 Gaussian exploration noise를 더한다. 
  - R frame skip 사용
- Planning = cross entropy method(CEM)  
  ![image](https://user-images.githubusercontent.com/46081019/59934389-5f7bfb00-9486-11e9-9c0a-603a1c101a58.png)  
  - Plannet은 explicit 'policy network'는 사용하지 않는다.
  - 대신 cross entropy method를 사용한다. 
  처음 본 알고리즘인데, 'population'-based optimization 기법으로써 objective를 maximize하는 action sequence를 배운다고 한다. 
  - "Action sequence"에 Diagonal Gaussian Belief with mean 0를 할당하고 이에 맞춰 처음 J posible sequence을 샘플링한다.
  - 그 다음 이를 전체 Reward 관점에서 평가, 정렬한다. J개 중 상위 K개의 seq를 뽑는다.
    - 이때 각 sequence에 따른 예상 결과(state)는 기존에 배운 state-space model을 활용한다.
  - 이 seq에 따라 action gaussian belief를 수정한다. 이를 *optimization* iter만큼 반복한다. 
  - 이 때, next observation을 받은 후에는 다시 action의 belief를 $$N(0, I)$$로 리셋하여 local optima에 빠지지 않도록 한다. 
  
  
**1. Recurrent State Space Model**  
  
- "Recurrent State Space Model은 non-linear Kalman Filter 혹은 sequential VAE로 볼 수 있다."
  - 이는 기존의 state space model 연구들에서 많이 다뤄진 내용이다. 
  - 여기서는 그런 다양, 비슷한 모델들 외에 근본적인 질문을 던진다; 
  - **BOTH Stochastic and deterministic paths in the transition model are crucial for successful planning**  
  - ![image](https://user-images.githubusercontent.com/46081019/59934897-91da2800-9487-11e9-9fc3-f86b77ef8cad.png)  
    - 기존 deterministic SSM: RNN
    - stochastic SSM: Gaussian State-Space Model.
       - Transition model: Gaussian with mean and variance parameterized by a feed-forward neural network
       - Observation model : Gaussian with mean parameterized by a deconvolutional neural network and identity covariance
       - Reward model: scalar Gaussian with mean parameterized by a feed-forward neural network and unit variance
    - 이 때 true state space는 알 수 없으므로, VAE를 활용하며 loss는 다음과 같다. 
       - ![image](https://user-images.githubusercontent.com/46081019/59935797-91429100-9489-11e9-8526-d4d0f3fdbb8f.png)  
  - 두 state space model은 장단점이 있다.
    - Purely stochastic transition: Difficult for the transition model to reliably remember information for multiple timesteps
    - 물론 필요한 경우 variance를 아예 0으로 만들면 deterministic하겠지만, optimization 과정에서 이는 매우 찾기 어려울 것이다.
    - 반면 deterministic은 확실히 기억은 할 수 있지만, stochastic noise에 취약할 것이다.
  - 따라서 논문은 deterministic sequence와 stochastic sequence 정보를 모두 갖는 것이 유리하다고 주장한다.
    - deterministic $$h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$$
    - stochastic $$s_t \sim p(s_t \mid h_t)$$
    - observation $$o_t \sim p(o_t \mid h_t, s_t)$$
    - Reward $$r_t \sim p(r_t \mid h_t, s_t)$$  
  - **이는 model의 각 $$state_t$$를 stochastic part와 deterministic part로 나눠 인코딩했다고 할 수 있다.**  
  
**2. Latent Overshooting**  
- Latent Overshooting은 [reddit discussion](https://www.reddit.com/r/MachineLearning/comments/aqzll1/r_planet_a_deep_planning_network_for/)에서도 다뤄지고 있다. 
  - 개인적으로 성능과 관계 없이 논문의 핵심 아이디어 중 하나라고 생각한다.
- Latent space가 multi-step prediction을 위한 정보를 갖도록 강제하는 것은 많은 SSM의 숙제이다.
  - **Latent overshooting은 VAE를 학습시키는 데에 있어 KL-divergence regularizer가 단지 one-step prior에만 의존하는 것을 개선하고자 한다.** 
  - 이는 반대로 얘기하면 학습 과정에서 gradient가 근시적 관점으로만 전달된다는 것이다. 
  - 따라서 이러한 limited capacity를 극복하고, multi-step prediction에 관련된 prior를 loss에 추가함으로써 전체적으로 latent

**3. Results**  

