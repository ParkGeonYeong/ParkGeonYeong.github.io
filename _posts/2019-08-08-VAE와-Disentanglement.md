---
title: 'VAE와 Disentanglement' 
use_math: true
classes: wide
layout: single
---

본 자료는 다음을 주로 참고했습니다.   
- [Beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)
- [Understanding disentangling in beta-VAE](https://arxiv.org/abs/1804.03599)
- [ELBO surgery](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf)
- [Beta-TCVAE](https://arxiv.org/abs/1802.04942)
  
  
VAE의 등장으로 latent variable의 posterior distribution을 근사할 수 있게 됨으로써 이제 관심사는 VAE을 통해 보다 '해석 가능한 분포'를 얻는 쪽으로 진화되어 왔다. 이를 위해서 representation learning, 즉 latent variable을 disentangling하려는 연구들이 등장했다. 여기서는 disentangling이 무엇인지, 그리고 disentangling을 위한 방법론에는 어떤 것들이 있는지 간단히 알아본다.
  
**0. Beta-VAE**  
**0.1. Understanding disentangling in beta-VAE**  
**1. Decompose prior loss of ELBO**  
**2. Beta-TCVAE**  
