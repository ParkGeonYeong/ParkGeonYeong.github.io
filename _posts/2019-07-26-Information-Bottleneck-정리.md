--- 
title: 'Information Bottleneck 정리' 
use_math: true
classes: wide
layout: single
---
  
  
본 자료는 다음을 참고했습니다.  
- [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057)
- [Deep Learning and the Information Bottleneck principle](https://arxiv.org/abs/1503.02406) 
- [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
- [InfoBot: Transfer and Exploration via the Information Bottleneck](https://arxiv.org/abs/1901.10902)
  
Information Bottleneck 이론은 Naftali Tishby 교수님이 2000년대 초 처음 주장하신 이론으로, 신호 Y에 대한 X의 정보를 최대한 보존하면서, X 자체는 최대한 압축하는 방법론이다. 몇 년 전부터는 functional space를 최소화하면서도 underfitting 없이 주어진 데이터를 잘 설명하는 딥 러닝 이론으로 떠오르고 있다. Tishby 교수님의 일련의 라이팅을 따라가다 보면 굉장히 얻을 것도 많고 딥 러닝에 대한 보다 큰 그림을 그려나갈 수 있다. 그리고 Deep Variational Information Bottleneck의 등장으로 최근에는 실제 training loss에도 활용되고 있다. 여기서는 original paper를 시작으로 Deep Variational Information Bottleneck과 그 응용 사례를 살펴본다.
  
**0. The information Bottleneck Method**  
- **IB Formalized the problem that finding a short code for X that preserves the maxmium information about Y through a 'bottleneck'**  
- Information Bottleneck은 결국 신호의 압축률과 정보 손실 간의 trade-off를 푸는 문제이다. 
  - 이는 "rate distortion theory"인데, rate는 압축률 또는 signal representation size이다. 
- 이때 신호의 distortion을 재는 것은 결국 신호의 meaningful 혹은 relevant feature을 알아야 한다는 뜻이고, 이는 쉽지 않다.
  - 원래 IB 이론에서는 우선 이 true distortion을 $$d(x, \tilde{x})$$으로 가정한다.
- **이후 이 "right distortion"을 대체하기 위해, Y라는 신호를 도입한다.**
  - 단 이때 Y는 반드시 X와 dependent하다
  - Positive joint distribution $$p(x, y)$$가 존재한다는 가정 하에, 
  Y와의 relevant information을 보존하는 것으로 distortion을 간접적으로 측정하는 것이다. 
  - **이 방식은 이후에 흥미로운 결과로 이어지는데, 우리가 가정한 $$d(x, \tilde{x})$$이 실제로는 $$D_{KL}{[p(y \mid x) \mid p(y \mid \tilde{x})]}$$이라는 것을 얻는다.**   
- $$\tilde{x}$$는 주어진 신호 x의 stochastical mapped code이다. 데이터셋 X를 soft-partitioning(혹은 quantizing)하는 codebook이라고 할 수 있다. 
- 이 코드북이 실제로 X를 잘 mapping하고 있는지 확인하기 위해서 두 가지 요소가 필요하다. 
  - 첫 번째는 앞서 언급한 신호의 압축률, 즉 X와 $$\tilde{X}$$ 사이의 mutual information $$I(X; \tilde{X})$$이다.
  - $$\tilde{x}$$이 원본 신호 대비 얼마나 압축되었는지, 혹은 independent한지를 의미한다. 
- 그러나 단순히 signal rate만 고려하면 $$X$$의 유의미한 feature 혹은 정보가 대부분 손실될 수 있다. 
  - 따라서 앞서 언급한, expected distrotion을 측정해야 한다. 
  - $$E[d(x, \tilde{x})_{p(x, \tilde{x})}] = \sum \sum p(x, \tilde{x})d(x, \tilde{x})$$ 
- 이제 distortion을 조건부 제약 조건으로 생각하고, $$I(X; \tilde{X})$$를 최소화해야한다. 
  - Rate distortion function $$R(D) = min_{p(\tilde{x} \mid x): E[d(x, \tilde{x})]<D} I(X; \tilde{X})$$
- 이는 lagrangian multiplier $$\beta$$를 도입하여 다음과 같이 쓸 수 있다. 
- **$$L[p(\tilde{x} \mid x)] = I(X; \tilde{X}) + \beta<d(x, \tilde{x})>$$**
  - Beta는 positive하다. 
- 위 식을 analytical하게 풀어 최적의 quantizing codebook $$p(\tilde{x} \mid x)$$를 구할 수 있다.    
  - ![image](https://user-images.githubusercontent.com/46081019/61927266-2405ac80-afaf-11e9-83d9-521d8bff3433.png)  
  - 증명은 편미분의 반복으로써 생략한다. 
- 눈여겨 볼 점이 두 가지인데, 우선 **true distortion $$d$$를 측정**하기 어렵고, **$$p(\tilde{x})$$와 $$p(\tilde{x} \mid x)$$가 서로 entangle**되어있다. 
  - 두 번째 문제의 경우 converging iterative algorithm을 통해 두 분포를 풀어낼 수 있다. 
  - ![image](https://user-images.githubusercontent.com/46081019/61927677-a6db3700-afb0-11e9-8708-c1147d8831a8.png)  
- 여기에는 중요한 Lemma가 쓰이는데, mutual information $$I(x; \tilde{x})$$을 최소화시키는 분포 $$p(\tilde{x})$$가 marginalized 분포라는 점이다. 
  - ![image](https://user-images.githubusercontent.com/46081019/61927888-85c71600-afb1-11e9-96af-85ad9b528e32.png)  
- 즉 위에서 찾은 최적의 $$p(\tilde{x} \mid x)$$와, marginal distribution $$p(\tilde{x})$$을 iterative하게 구하면 수렴한다.
  
  
- 이제 첫 번째 문제를 풀기 위해 처음 언급했던 dependent variable $$y$$를 도입한다. 
- 처음 구했던 information bottleneck은 다음 식으로 변경된다. 
- **$$L[p(\tilde{x} \mid x)] = I(X; \tilde{X}) - \beta{I(\tilde{x}; Y)}$$**
  - X와 Compressed X는 최대한 independent하게, Compressed X와 Y는 최대한 dependent하게
- 이제 여기서도 analytical하게 $$p(\tilde{x} \mid x)$$를 구해보자.
  - ![image](https://user-images.githubusercontent.com/46081019/61928946-0b989080-afb5-11e9-89f7-0bcd279a3bec.png)  
  - 이때 $$x, \tilde{x}, y$$ 사이에 markov chain을 가정하기 때문에 위와 같은 식을 얻을 수 있다. 
  - **처음 distortion function d를 가정하고 얻었던 식과 비교 했을때, $$d(x, \tilde{x}) = D_{KL}[p(y \mid x) \mid p(y \mid \tilde{x})]임을 알 수 있다**
    - 이는 Distortion에 어떠한 가정을 하지 않은 채 얻은 결과이다. 즉 Distortion은 $$\tilde{x}$$가 true joint disribution $$p(x, y)$$를 얼마나 잘 설명하는지에 대한 값이라고 할 수 있다. 
  - **우리가 데이터를 많이 갖는 것이 meaningful feature extraction에 얼마나 중요한 지를 보여주고 있다.**
    - True joint distribution을 sparse하게 estimate할 경우, $$\tilde{x}$$가 실제 분포와는 무의미한 정보를 인코딩하게 될 수도 있다. 
    - 따라서 데이터의 질, 그리고 양이 rate distortion theory 관점에서 중요하다는 걸 알 수 있다. 
  - **또 다른 차이점은, 이제 $$p(\tilde{x} \mid x)$$가 $$p(\tilde{x})$$ 외에도 $$p(y \mid \tilde{x})$$에 dependent하다는 것이다.** 
    - 기존에는 $$x->\tilde{x}$$의 soft partitioning에만 관심이 있었다면, 이제는 y까지 고려하여 meaningful한 $$\tilde{x}$$를 찾아야 한다.
  
  
- 위의 식을 보면 결국 우리가 구해야 하는 것은 $$p(\tilde{x} \mid x), p(\tilde{x}), p(y \mid \tilde{x})$$이다. 
  - 우리가 처음 가정했던 markovian chain에서 뽑을 수 있는 분포와 동일하다. 
- 결국 이 세 분포도 converging iterative algorithm을 통해 풀 수 있다. 
- **결론적으로 원래 information bottleneck theory는 analytical한 분석을 통해 most efficient informative features (or approximate minimal sufficient statistics)을 뽑을 수 있는 이론이다.**
  - 그러나 우리는 아쉽게도 데이터의 true joint distribution $$p(x, y)$$을 알지 못한다.
  - 이는 곧 우리가 empirical estimate을 통해 위 문제를 풀어야 한다는 말이다.
  - 그리고 tishby 교수님은 딥러닝이 복잡한 데이터에 대해서도 이 empirical estimate을 잘 한다고 주장한다.

**1. Explaining Deep Neural Networks by IB**  

**2. Deep Variational Information Bottleneck**  

**2.1. Applications in RL: InfoBot**  
