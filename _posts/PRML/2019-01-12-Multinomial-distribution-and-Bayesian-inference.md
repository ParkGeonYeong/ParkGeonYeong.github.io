---
title : "Multinomial Distribution and Bayesian Inference"
use_math : true
layout: single
classes: wide
---

**0. 동전 던지기에서 앞면이 나올 확률은?**
모두가 알고 있듯이 0.5이다.  

**0.1. 동전을 5번 던졌는데 모두 앞면이 나왔더라도 앞면이 나올 확률을 0.5라고 할 수 있는가?**
동전을 3번 던졌던, 5번 던졌던 우리는 결과와 상관없이 앞면이 나올 확률을 0.5라고 알고 있다.  이는 이미 우리가 **동전 앞/뒷면 변수**에 대한 사전 확률을 강하게 가정하고 있기 떄문이다.  
그렇다면 어떤 식으로 이런 사전 확률을 가정하고, 또 갱신하는 것일까.
Binomial/Multinomial Distribution과 베타 분포 등을 통해 이를 알아보자.

**1. Binomial Distribution**
이항 분포에서 동전의 앞면이 나올 확률을 $$\ㅡ
