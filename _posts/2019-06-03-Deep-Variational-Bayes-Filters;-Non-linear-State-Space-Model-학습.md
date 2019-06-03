---
title: "Deep Variational Bayes Filters; Non-linear State Space Model 학습"
use_math: true
classes: wide
layout: single
---
  
  
다음 논문을 정리했습니다. 개인적 해석이 다수 포함되어 있습니다. 
- [DEEP VARIATIONAL BAYES FILTERS: UNSUPERVISED LEARNING OF STATE SPACE MODELS FROM RAW DATA](https://arxiv.org/pdf/1605.06432.pdf)  
  
이전 포스트와 연관이 있습니다.  
- [Variational Inference 포스트](https://parkgeonyeong.github.io/Gaussian-Process%EC%99%80-Variational-Inference/)
- [State-Space Model](https://parkgeonyeong.github.io/Model-based-Planning-and-some-recent-works/)
  
  
<요약>  
논문을 한 문장으로 요약하자면 **State-Space Model Learning using Temporal(or transitional) informative Latent variable**
이라 하겠다. State-Space Model(SSM)은 환경과 객체의 dynamics을 표현하는 모델이다. 
논문에서 제기하는 문제는, 기존 state-space model이 지나치게 "reconstruction of observation"에 집중하여 proper future prediction에 대한 
정보를 잃고 있다는 점이다. 따라서 곧 SSM이 one-step transition만을 올바르게 예측하게 된다. 
이에 저자는 $$z_{t+1}=f(z_t, u_t, \beta_t)$$으로 latent dynamics을 sequential dependent하게 만들어 
latent variable이 physical dynamics을 잘 인코딩하도록 유도한다. ($$u_t: action, \beta_t: transition parameters$$)
그 결과 $$z_t$$는 모델의 inherent physical manifold를 잘 학습하게 된다.
  
  
**0. Introduction**  
- State space model : Powerful tool to analyze and control the dynamics
- State space model을 활용하여 probabilistic한 환경에서 $$x_{1:T}$$에 대한 추론을 할 수 있다.
  - $$p(x_{1:T} \mid u_{1:T}) = \int{p(x_{1:T} \mid z_{1:T}, u_{1:T})p(z_{1:T} \mid u_{1:T})dz_{1:T}}  (1)$$
  - $$z$$은 latent variable
  - $$x$$를 estimate하는 부분은 emission, $$z$$를 estimate(assume we are estimating true latent variable)하는 부분은 transition
- High-dimensional domain에서 이 두가지를 동시에 같이 잘 하기는 어렵기 때문에, 효율적인 latent system identification이 중요하다
  - In here: enforces the latent state-space model assumption, allowing for reliable *system identification* and *long-term prediction*
- 본 논문에서 위의 SSM을 변형한 방식: Transition Parameter $$\beta$$를 도입
  - Latent variable이 dynamics을 잘 인코딩하지 못 할 경우 미래 예측이 diverge할 수 있다.
  - 따라서 transition에 대한 정보를 갖고 있는 $$\beta$$를 도입하여 일종의 bayesian regularizing prior로 활용한다.
  - $$\int\int{p(x_{1:T} \mid z_{1:T}, u_{1:T})p(z_{1:T} \mid u_{1:T}, \beta_{1:T})p(\beta_{1:T})dz_{1:T}d\beta_{1:T}}$$
  - 이때 기타 SSM과 동일하게 $$z_t$$는 markovian을 가정하며, $$u_{1:T}$$ 의존도는 없다고 가정하고 생략할 수 있다.
  - Linear한 Gaussian state transition model을 가정하면 kalman 등의 classic한 방법으로 풀 수 있는 문제지만, 
  High-dimensional domain에서는 그렇지 않다.
- 논문에서는 SSM이 "good compression only" (like VAE)와 같은 SSM에 빠지지 않아야 한다고 주장한다. 
  - Cost of decreasing the reconstruction을 감수하고 prediction에 더 집중해야 한다. 
  - **We force the latent space to fit the transition--reversing the direction, and thus achieving the state space model assumptions and full information in the latent states**
  
  
**1. Model structure**   
- 논문에서 dynamics을 위해 latent space를 학습시킨 원리는 다음 한 문장으로 설명 가능하다.
  - **We establish gradient paths through transitions over time so that the transition becomes the driving factor for shaping the latent space**
- $$z_{t+1}=f(z_t, u_t, \beta_t)$$
  - 그래프로 표현하면 다음과 같다.  
  - $$\beta$$는 현재 환경의 전반적 transition dynamic에 대한 meaningful prior이며, 위 그래프처럼 $$v_t, w_t$$로 나뉜다. 
    - $$w_t$$는 현재 sample data에 dependent한, sample-specific noise 역할이다.
    - $$v_t$$는 sample에 independent한 universal transition parameter로, 개인적인 생각으로는 환경의 global한 특징 및 
    과거 observation과의 temporal한 correlation 정보를 포함하고 있는 것으로 보인다. 
  - 본 논문에서는 VAE의 인코더와 같이 기능하는 모델을 recognition model $$q$$이라 칭한다. 이때 $$q$$는 다음과 같이 factorize 가능하다.
    - $$q_\phi(\beta_{1:T} \mid x_{1:T})=q_\phi(w_{1:T} \mid x_{1:T})q_\phi(v_{1:T})$$
    - 즉 베타는 일종의 transition parameter, 혹은 another latent rather than z이라 할 수 있겠고 이는 $$x$$ dependent와 independent 파트로 나뉜다.
- $$z$$를 $$z, u, \beta$$로 표현할 수 있으므로 위의 (2)는 다음과 같이 바뀐다. 
  - 따라서 Variational Lower Bound는 다음과 같다. 
- 전체 모델 구조는 다음과 같다.  
  - 가장 중요한 최종 latent variable은 다음과 같이 weight를 부여 받아 계산된다. 
  - $$z_{t+1}=A_{t}z_{t}+B_{t}u_{t}+C_{t}w_{t}$$
  - $$A, B, C$$는 기존의 latent variable(=environment가 어디 정도인지 가늠하는 정보), 현재 액션, 현재 observation과 액션을 통해 인코딩된 transition parameter 세 가지 정보가 조합되는 가중치이다. 이때 이 가중치는 $$q_\phi(v_t)$$에서 나온 것으로, 즉 현재 sample과는 무관하게 
  $$q_\phi$$가 universal environment dynamics을 파악해 감에 따라서 적절하게 조정될 것이다.   
  - 단 각 $$(i)$$ matrices가 "조합"되어 최종 가중치를 만드는 것은 $$z_t, u_t$$에 dependent하다
- 요약하자면 latent variable $$z_{t+1}$$은 현재 액션 $$u_t$$ 외에도 직전 latent variable $$z_t$$와, 
transition parameter $$w$$에 의존적이다. 
또한 이 세 가지 정보가 각자 가중치를 부여 받을 때 "각 observation at timepoint"와 "universal transition parameter"에 모두 영향을 받는다.
  - 따라서 end-to-end training 과정을 거치면 latent variable이 Model dynamics에 민감해진다.
  
  
**2. Experiments results**  
- 실험은 거의 proof of concept 수준으로 진행되었으며, 결과 역시 깔끔하다.
- Non-markovian pendulum의 각, 각속도를 observation 삼아 학습시켰다.
  - z dimension은 visualize를 위해 3으로 고정했다. 
  - $$z_1$$ 축을 따라 angle velocity가, $$(z_0, z_2)$$ plane을 따라 angle이 인코딩되었다. 
  - 반면 Deep Kalman filter는 각속도 인코딩에 실패했다.
  - 학습한 latent variable manifold를 따라 latent variable이 나선형으로 walking하는 것을 알 수 있다.
