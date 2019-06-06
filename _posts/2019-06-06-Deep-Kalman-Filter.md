---
title: "Deep Kalman Filter"
use_math: true
classes: wide
layout: single
---
  
  
다음 논문을 정리했습니다.
- [Deep Kalman Filter](https://arxiv.org/abs/1511.05121)  
  
이전 포스트와 연관이 있습니다.  
- [Variational Inference 포스트](https://parkgeonyeong.github.io/Gaussian-Process%EC%99%80-Variational-Inference/)
- [State-Space Model](https://parkgeonyeong.github.io/Model-based-Planning-and-some-recent-works/)
- [Deep Variational Bayes Filter](https://parkgeonyeong.github.io/Deep-Variational-Bayes-Filters;-Non-linear-State-Space-Model-%ED%95%99%EC%8A%B5/)
  
  
**요약**  
핵심은 굉장히 간단하다. Neural Network와 Variational Inference를 통해 classical 'linear' kalman filter를 개선한 논문이다. 

**0. Introduction**  
- Kalman Filters
  - 본 논문을 비롯해 다양한 planning or prediction domains(Video Generation, Motion planning, Model-based learning ...)은 
  Generative model을 적극적으로 사용한다. 
    - Generative model을 통해 일련의 observations $$x_t$$과 actions$$u_t$$을 예측하는 것이 목적이다. 
    - 각 prediction은 해당 sequential information을 배우며 진화하는 latent variable에 의해 생성된다.
  - 전통적인 control theory, robotics 분야에서는 이러한 latent variable dynamics을 2단계(선형)로 나누어 설명했다.
    - $$z_t = G_{t}z_{t-1} + B_{t}u_{t-1}+\epsilon_t$$, $$x_t = F_{t}z_{t} + \eta_t$$
      - $$\epsilon$$은 zero-mean gaussian noise, $$u$$는 action, $$G, B, F$$는 latent, action, reconstruction transition matrix
    - 이때 각 transition matrix는 linear transormation으로, 복잡한 non-linear dynamics 표현에 한계가 있었다.
    - 이에 본 논문은 이를 non-linear neural nets으로 대체하고자 한다.
  
**1. Model**  
  - Non-linear한 observation sequence와 action을 인코딩하기 위해 latent variable에 variational inference가 사용되었다.
    - ![image](https://user-images.githubusercontent.com/46081019/59028632-fc3f7580-8896-11e9-82fd-3145b576ad45.png)  
    - $$q$$는 VAE에서의 인코더와 비슷하다.
    - 기존의 kalman filter가 linear projection으로 latent variable을 얻어냈다면, 여기서는 $$q$$를 이용한 distribution을 얻어낸다. 
    - $$z_1 \sim N(\mu_0, \sum_0)$$
    - $$z_t \sim N(G_\alpha(z_{t-1}, u_{t-1}, \delta_t), S_\beta(z_{t-1}, u_{t-1}, \delta_t))$$
    - $$x_t \sim \prod(F_k(z_t))$$
    - 이때 $$G, S, F$$는 neural network이다.
      - $$Z$$의 prior distribution으로 multivariate gaussian을 사용하였고, diagonal covariance matrix을 사용한다.
      - 또한 $$z_t$$가 $$z_{t-1}$$ dependent하기 때문에 $$q$$로 MLP, RNN, BiRNN을 시험해 보았고 
      예상대로 BiRNN이 가장 좋은 성능을 보였다고 한다.
      - ![image](https://user-images.githubusercontent.com/46081019/59029625-b33cf080-8899-11e9-9707-1c0ae579f610.png)  
  - 메인 Variational Lower Bound는 다음과 같다.
    - ![image](https://user-images.githubusercontent.com/46081019/59029103-5856c980-8898-11e9-8139-b79e9a64955c.png)  
    - 이때 벡터 $$z$$을 시간에 대해 factorize하여, 새로운 $$z_t$$ 들에 대해 다시 KL-div를 쓸 수 있다. 
    - ![image](https://user-images.githubusercontent.com/46081019/59029268-cbf8d680-8898-11e9-8fac-ae97cdb675b7.png)  
    - Prior p의 $$z$$ 역시 시간이 지남에 따라 갱신된다.
  - 시간에 대해 autogressive하게 누적되었을 뿐 기존의 알고 있었던 ELBO와 동일한 형태이며 따라서 end-to-end로 학습이 가능하다.
    - 즉 매 time point마다 새로운 $$z$$을 샘플링하고, 이를 기반으로 다시 $$x$$를 얻은 다음 Loss를 업데이트한다.
    - 각 $$z_t$$ (from posterior distribution)은 markov하다는 가정이 들어간다. 
    이때 각 $$z_t$$는 이전 모든 $$x$$에 대한 abstract information을 갖고 있다는 가정이 필요한 것으로 보인다.
    
    
    
**2. Experiment**  
  ![image](https://user-images.githubusercontent.com/46081019/59030206-4aef0e80-889b-11e9-852d-a1467d29e2f6.png)  
