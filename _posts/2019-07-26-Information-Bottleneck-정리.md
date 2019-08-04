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
  - **처음 distortion function d를 가정하고 얻었던 식과 비교 했을때, $$d(x, \tilde{x}) = D_{KL}[p(y \mid x) \mid p(y \mid \tilde{x})]$$임을 알 수 있다**
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
딥러닝의 부상과 맞물려 tishby 교수님은 딥러닝과 IB를 관련 짓는 두 편의 논문을 쓰셨다. [Deep Learning and the Information Bottleneck principle](https://arxiv.org/abs/1503.02406)은 뉴럴 네트워크의 각 layer를 Information Bottleneck의 markovian chain과 연결짓는 이론적 연구이고, [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)는 뉴럴 네트워크의 SGD optimization이 실제로 뉴럴 네트워크를 optimal IB로 보낸다는 내용이다. 
  
데이터 분포를 (X, Y)으로 생각하면, 뉴럴 네트워크는 Y -> X -> h1 -> ... -> h의 markovian chain이라고 할 수 있다. 이를 통해 각 layer의 인풋이 직전 레이어의 확률적 output에만 의존하기 때문이다. 이는 곧 *각 layer*에 대해서 IB limit(or distortion)을 정의할 수 있다.   
$$D_{IB} = I(h_{i-1}; h_i) + \beta{I(Y; h_{i-1} \mid h_i)}$$  
이를 그래프로 표현하면 다음과 같다.  
![image](https://user-images.githubusercontent.com/46081019/62421936-18b82c80-b6e5-11e9-9d5c-e81d11f3f25d.png)   
그래프에서 black line은 신호의 각 description length(R)당 정해진 optimal IB limit이다. Discription Length가 길어질 수록 IB는 낮아지지만, discrption cost는 늘어날 것이다. Green Line으로 표시된 딥 뉴럴 네트워크 역시 Compression이 진행될수록 discrption length는 짧아지고, 그 대가로 IB는 상승하는 것을 알 수 있다. Blue line은 어떤 critical beta 값에 의해 나타날 수 있는 suboptimal bifurcation이다. Green Line에서도 압축이 진행됨에 따라 이 blue line과 만날 수 있는데, beta를 얼마나 조절했느냐에 따라 어디서 만나는지가 결정될 것이다. 가령 beta를 줄일 수록 압축이 더 진행되는 것이고, 따라서 green line의 위로 올라갈 수록 empirical하게 beta를 줄였다고 볼 수 있다. 그렇다면 beta 설정에 의한 trade-off의 optimal point는 과연 어떻게 결정할 수 있을까? 이는 유한한 data로 인한 finite sample bound (orange line)을 통해 확인할 수 있다; 신호의 복잡도는 최소화하면서 IB distortion은 너무 높이지 않는, 주황색 그래프의 극소점을 true trade off point라고 할 수 있다. 현재 최종 압축 형태인 $$\hat{y}$$와의 거리를 각각 $$\triangle{C}, \triangle{G}$$으로 표현할 수 있다. 참고로 이 유한한 데이터로 인한 finite bound는 다음 두 empirical equation을 통해 구할 수 있다고 한다; 신기한 점은 empirical information estimation의 boundary가 hidden state의 cardinality $$K=\mid \hat{X} \mid$$에만 의존한다는 점이다; 즉 다시 말해 데이터의 inherent한 복잡도 $$\mid \hat{X} \mid$$에는 큰 영향을 받지 않으며 이는 뉴럴 네트워크가 복잡한 데이터도 튜닝에 의해 학습할 수 있음을 의미한다. 
![image](https://user-images.githubusercontent.com/46081019/62422064-7a799600-b6e7-11e9-80a1-bbf558c1877a.png)  
![image](https://user-images.githubusercontent.com/46081019/62422072-8d8c6600-b6e7-11e9-9e57-239f95c1f9bc.png)  
  
   
Tishby 교수님은 이론적인 접근을 넘어 실험적으로 딥러닝이 어떻게 information channel 혹은 bottleneck과 비슷하게 perform하는지를 보였다.   
[Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)에서 몇 가지 중요한 논의를 던졌는데, 
1. 뉴럴 네트워크를 학습시킬 때 training labels에 fitting되는 과정(Empirical Error Minimization)이 아니라 efficient representation을 향해 데이터를 압축시키는 과정(representation compression)에 epoch이 많이 소요된다.
2. 이 representation compression phase는 training error가 충분히 작아졌을 때 시작되며, 이는 mini-SGD로 인한 Noise의 개입으로 인해 이루어지는 random diffusion process이다. 
3. 수렴한 레이어들은 Ideal Information Bottleneck의 single point에 해당하는 경향을 보인다.
4. Hidden layer는 계산량을 빠르게 줄이는 데에 관여한다. 
  
관련 결과들을 살펴보면 다음과 같다.  
![image](https://user-images.githubusercontent.com/46081019/62422897-b1a27400-b6f4-11e9-9299-1628c3c28782.png)  
초기, 400epoch, 9000epoch 이후의 information plane. 전체 9000 epoch 중 400 epoch의 단계에서, 약간의 descption length를 댓가로 $$I(T; Y)$$가 많이 올라온 것을 알 수 있다. 이후 훨씬 긴 epoch에 대해서는 다시 descrption length를 줄이며 efficient representation의 학습이 진행됨을 알 수 있다.    
![image](https://user-images.githubusercontent.com/46081019/62423240-b9b0e280-b6f9-11e9-9a6b-669eafe3e3c7.png)  
왼쪽부터 5%, 45%, 80%의 데이터를 사용한 Training information plane. 데이터 양에 크게 상관 없이 ERM loss는 비슷한 속도로 줄어들지만, 이후 random diffusion phase에서 큰 차이가 난다. 이는 random diffusion phase가 overfitting와 직접적으로 연관된다는 점을 의미한다. 데이터가 작아도 joint distribution의 큰 흐름은 배울 수 있지만, 이후 generalization 단계에서는 stochastic한 representation을 학습하기 어렵다는 뜻이라 생각한다.   
  
![image](https://user-images.githubusercontent.com/46081019/62423324-87ec4b80-b6fa-11e9-98a5-6cf17f908335.png)   
언급한 ERM minimization phase, random diffusion phase가 어떻게 나뉘는지를 잘 보여주는 figure이다. X축이 log-scale에 유의하면, 초기 짧은 epoch 동안은 실선으로 표시된 Norm of average gradient가 지배적이지만 이후에는 norm이 줄어들고 Gradient가 보다 stochastic하게 발산한다. 이는 곧 학습 과정에서 **noise의 중요성**을 의미한다.   
**Mutual Information은 invertible transformation에 invariance하기 때문에, function complexity를 올바르게 측정하거나 고려하기 어렵다.** 즉 만약 어떤 invertible transformation f에 대해 $$y=f(x)$$으로 정의하면, $$I(x; y)$$와 f의 cardinality 정보는 무관하며 따라서 mutual information만으로 overfitting을 확인하기 어렵다. 그러나 만약 y와 x의 관계를 $$p(y \mid x)$$으로 stochastic하게 표현한다면, 즉 sigmoid, tanh 등의 non-linear probabilistic activation function을 활용한다면 joint distribution $$p(x, y) = p(y \mid x)p(x) = \frac{p(x)}{1-exp(y-wx+b)}$$ 등으로 표현할 수 있다. 즉 이 joint distribution을 학습하는 과정에서 비로소 w의 complexity가 중요해지며 mutual information에도 이러한 정보가 고려되게 된다. 미니 배치 $$p(x)$$의 stochasticity와 stochastic spread of sigmoid function으로 인해 function complexity에 sensitive한 학습이 가능한 것이다.   
    
   
![image](https://user-images.githubusercontent.com/46081019/62423753-9be67c00-b6ff-11e9-88aa-c6e4b4a7722c.png)  
앞서 본 데이터 크기에 대한 차이를 일반화한 6-layers information plane이다. Low layer일 수록 random initialization만으로 original dataset X에 대해 높은 정보를 유지하기 때문에 데이터 크기에 큰 영향을 받지 않으며, higher layer일 수록 training data가 많아짐에 따라 데이터셋중 relevant x를 학습하며 optimal information plane에 가까워짐을 알 수 있다. 데이터셋이 더 충분히 확보되면 relevant x feature가 더 다양해짐에 따라 $$I(T; X)$$가 조금 증가함을 알 수 있다. 

**종합하면 neural network는 stochastic representation 학습을 통해 overfitting에 강한 함수를 얻을 수 있고, 그 과정에서 data 자체의 Complexity에도 invariant하기 때문에 전반적으로 복잡한 데이터에 대한 low-dimensional projection을 잘 얻을 수 있다.** 
지금까지 살펴본 바로 information bottleneck은 이론적으로 탄탄하다는 장점이 있었지만, **실제 학습에 활용하기는 다소 어려운 점이 있었는데 이는 Hidden space z에 대한 mutual information을 뽑아내야 하기 때문이다. 그러나 이를 variational inference로 해결한 논문이 등장하면서, 이제 실제 neural network의 학습에도 information bottleneck 이론이 응용될 수 있게 되었다.**

**2. Deep Variational Information Bottleneck**  

**2.1. Applications in RL: InfoBot**  
