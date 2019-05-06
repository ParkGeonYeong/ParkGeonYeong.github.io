---
title: Gaussian Process과 Variational Inference 유도 및 응용(1)
use_math: true
classes: single
layout: wide
---

*Rough Note*
- GP: dispense with the parametric model and instead define a prior pr over 'functions' directly
  - 함수에 대해서 Non-parametric하다. 데이터셋을 기반으로 정의한다.
  - 반면 bayesian linear regression은 p를 정의했던 parametric model이다.
  - Uncountably infinite space of functions에서 어떻게 해야하는지 막막하지만, 
  for a finite training set에서 values of the function at the discrete set of input values xn만 고려하면 test set에서의 data를 얻을 수 있다.
  - BNN은 GP의 approximation이다. [논문 참고](https://openreview.net/pdf?id=B1EA-M-0Z)
- Linear regression with gaussian prior linear regression은 GP의 특수 케이스이다. 
  - noise를 더해야 한다.
  - Predictive distribution of new data point!
    - Parameteric view : regression의 w를 잡고 풀기
    - Function view : f*에 대해 바로 풀기 (conditional gaussian)

본 자료는 다음을 주로 참고하였습니다.  
- PRML Chap 6. Kernel Methods
- Edwith 최성준님 강의, Bayesian Neural Network
- [Weight Uncertainty in Neural Networks, 2015](https://arxiv.org/abs/1505.05424)
- [VIME, 2016](https://arxiv.org/abs/1605.09674)
- [TD-VAE, 2019](https://arxiv.org/abs/1806.03107)
**Introduction**  
Gaussian Process과 Variational Inference는 uncertainty in deep learning, variational auto-encoder, bayesian neural network 등의 이론적 기반이다.
머신 러닝의 확률론적인 적용에 관심이 있다면 반드시 탄탄하게 알아두어야 한다. 여기서는 Gaussian Process, Gaussian Process와 Variational Inference의 연관성, 
Bayesian neural network까지 이어서 설명하려 한다. 이 후 강화 학습 분야의 적용 예시을 알아보려 한다. 
Exploration, Curiosity 쪽 분야의 클래식한 논문인 VIME과 얼마 전 ICLR에 발표된 TD-VAE이다.
지금까지의 블로깅 중 가장 긴 시리즈가 될 것 같지만 꾸준히 써 봐야겠다.  

**0. Gaussian Process**
