---
title: Variational Auto Encoder(VAE)
use_math: true
classes: single
layout: wide
---

*이전 포스트와 이어집니다.*
- [Gaussian Process와 Variational Inference](https://parkgeonyeong.github.io/Gaussian-Process%EC%99%80-Variational-Inference/)  
*다음 자료를 주로 참고했습니다.*  
- [Original Paper](https://arxiv.org/pdf/1312.6114.pdf)
- [Jaejun Yoo님 블로그 포스트](http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html)  

출판 이후 생성 모델로 꾸준히 활용되고 있는 VAE이다. 
Variational inference, VAE, bayesian neural network 등은 모두 풀고자 하는 목표가 동일하다. 
이전 포스트에서 Variational Inference lower bound, 혹은 ELBO의 유도는 log likelihood $$logp(x)$$의 최대화(MLE)에서 시작했다. 
'Auto-encoder'라는 이름에서 알 수 있듯이, VAE는 self-supervised MLE 문제를 푸는 것이며, 
bayesian neural network는 보다 일반적인 training set에 대해 MLE를 푼다고 할 수 있다. 
여기서는 VAE의 의의, 수식적 전개와 모델 구조를 주로 알아보겠다. 

**0. 

The variational parameters φ are learned jointly with the generative model parameters θ.
