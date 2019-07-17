---
title: "Normalization 정리"
use_math: true
layout: single
classes: wide
---  
  
  
**본 자료는 다음을 주로 참고했습니다.**  
- [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) 
- [Weight Normalization](https://papers.nips.cc/paper/6114-weight-normalization-a-simple-reparameterization-to-accelerate-training-of-deep-neural-networks.pdf)
- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
- [How Batch Normalization Help Optimization?](https://arxiv.org/pdf/1805.11604.pdf)
- [Weight Standardization](https://arxiv.org/pdf/1903.10520.pdf)
- [Fisher Information](https://web.stanford.edu/class/stats311/Lectures/lec-09.pdf) 
  
  
Batch Normaliazation 등장 이후 Activation, Weight, Layer 단위에서의 Normalization은 이제 빼먹을 수 없는 중요한 요소가 되었다. 
대부분의 모델 구조에서 사용되는 Normalization이지만, 그 이론적 기반은 아직도 논의 중이며 
최근에는 loss, gradient landscape의 smoothing이 주목 받고 있다. 
따라서 Normalization paper에서 해결하고자 하는 문제와 수학적 테크닉은 근본적으로 대부분 비슷하다. 
여기서는 실험보다는 normalization에 사용되는 이론적 기반(lipschitzness, fisher information, Hessian 등)을 위주로 지난 벤치마크 논문들을 되돌아 본다. 
  
**0. Batch Normalization**  
많이 알려져 있듯이 Original Batch normalization 논문이 제기했던 문제는 internal covariate shift의 해결이다. 
딥러닝에서 weight update는 upstream gradient와 previous layer output이 결합되어 이루어진다. 
이때 previous output distribution이 학습 도중 계속해서 변할 경우 gradient의 update 역시 계속해서 그 방향성이 바뀔 수 밖에 없다. 
만약 sigmoid, tanh등 non-linear saturation region이 있는 activation function을 사용할 경우 이전 layer의 distribution은 
더 자주, 급격하게 변한다. 이는 input의 standardization으로는 해결할 수 없는 문제인데, input scale이 standardize되더라도 그 효과는 
hidden layer를 거치면서 소멸되기 때문이다. 
  
  
그렇다고 해서 각 hidden layer 이후에 naive한 normalization layer(not learnable)를 추가하는 경우 
오히려 이전 layer의 gradient update 효과를 무시해버리게 된다. 이전 $$l_i$$ layer의 input을 u, bias를 b라 하자. 
이때 naive normalization layer $$l_N$$을 추가하면, 이전 layer의 bias update를 $$\triangle b$$이라 할 때 
$$l_N = l_i(u) - E[l_i(u)] = u + b + \triangle b - E(u + b + \triangle b) = u - E[u]$$  
으로 update 효과가 사라지게 된다. 
이는 standard deviation scaling이 추가되면 더 심각한 문제가 된다. 
  
즉 normalization할 때 이전 layer의 two moments(expectation, variance)에 대한 gradient도 고려해야 한다. 
Batch Normalization에서는 이를 parameter $$\gamma, \beta$$을 도입하여 해결한다.   
단 whole training data의 expectation, variance에 대해 gradient를 고려하게 되면 계산량이 너무 많기 때문에 'batch'을 도입하는 것이다. 
따라서 test에서는 각 batch(=sampled distribution)의 expectation, variance를 이용하여 전체 데이터에 대한 true estimation을 구해 쓴다. 
이 때 sampled distribution의 unbiased variance는 분모에 $$N$$이 아닌 $$N-1$$이 들어가기 때문에 각 batch의 $$\sigma$$을 무작정 평균내면 안된다. 
  
Batch Normalization에서는 데이터를 feature-wise로 normalize한다. 즉 CNN에서는 kernel의 각 joint spatial location에 대해 normalize함으로써 convolution property를 유지하고 spatial information을 그대로 전달할 수 있다. 
  
**논문을 읽으면서 internal covariate shift에 대한 논의가 부족하다고 생각했다. 직관적으로 internal covariate shift는 모델이 sharp local minima에 취약한 이유가 될 수 있다고 생각했지만, 이론적으로 이에 대한 수식이나 실험이 뒷받침되지는 않았다.** 오히려 이후에 살펴볼 "How Batch Normalization Help Optimization?"에서 이를 엄밀하게 검증하는데, internal covariate shift(ICS)가 모델의 성능에 방해가 되지 않으며 심지어 BN이 ICS를 줄이지도 않는다고 주장한다. **오히려 BN 논문에서 보다 매력적으로 느낀 부분은 BN이 parameter growth를 제한할 수 있다는 점과 
layer jacobian이 singular value를 모두 1에 가깝게 갖도록 유도한다는 점이다. **
딥러닝 학습을 하다 보면 파라미터가 점점 커져서 그 범위가 튀어버릴 때가 많은데, activation output을 강제로 normalize하면 이런 explosion이 발생활 확률을 현저히 낮춰 준다. 이는 이후 논문들에서 loss와 gradient의 lipschitzness를 구할 때 기본이 되는 가정으로, batch normalization도 이를 implicit하게 갖고 있지만 논문에서 수식적으로 보이지 않은 것으로 보인다. 
또한 $$z=Fx$$에서 layer $$F$$를 단순히 linear transformation으로 가정하고, normalized x와 z를 각각 uncorrelated gaussian으로 가정할 때 $$I = cov(z) = Fxx^{T}F^{T} = FF^T$$으로, F의 singular value는 1이 되어 gradient의 magnitude를 제한한다. 물론 실제로는 non-linear transformation이지만 BN이 학습을 안정화시킨다는 간접적인 증거이다. 이런 내용과 관련된 논의는 2. How Batch Normalization Help Optimization?에서 알아본다.  
  
  
**1. Weight Normalization and Layer Normalization**  
Batch Normalization 이후 나온 많은 normalization paper들이 ICS 자체보다는 gradient와 learning rate의 안정화에 집중했다. 그 중에서도 weight normalization과 layer normalization은 batch 단위가 아닌 weight와 layer 단위에서 normalize함으로써 batch의 statistics 추정에서 오는 noise를 최소화했고 RNN의 적용 가능성을 높였다. 
  
![BTQ0zj5](https://user-images.githubusercontent.com/46081019/60778673-dc8bbd80-a172-11e9-8c88-1afeeb4c5fcc.png)  
Weight Normalization은 gradient의 whietening, layer normaliztion은 fisher information의 stabilize라고 요약할 수 있겠다. 
이 두 논문을 보다 쉽게 이해하기 위해서 우선 fisher information을 살펴 본다.   
**1.1 Fisher Information**  

**1.2 Weight Normalization**  
Gradient의 whietening은 dimension 간의 correlation을 줄임으로써 업데이트 과정을 '쉽고 예측 가능하게' 만들어 준다고 생각한다. 
Weight Normalization 이전에는 gradient에 fisher information matrix의 approximate inverse를 곱하거나, fisher information matrix을 diagonalize하는 식의 연구가 있었다고 한다. Weight normalization은 fisher information matrix을 explicit하게 다룬다기 보다, weight space를 reparameterize하는 방식이다. 
  
$$y=\phi(w*x+b)$$에서 $$w=\frac{g}{\mid\mid v \mid\mid}v$$으로 reparameterize한다. 이렇게 weight의 direction과 magnitude를 분리함으로써 convergence의 속도를 높일 수 있다. 우선 각 g, v에 대한 gradient를 구하면 다음과 같다. 
![image](https://user-images.githubusercontent.com/46081019/60780608-2330e600-a17a-11e9-8853-ce07a74398c8.png)  
이때 v gradient를 다시 쓰면 다음과 같다.   
![image](https://user-images.githubusercontent.com/46081019/60780632-3c399700-a17a-11e9-872c-ec87afdf26ef.png)   
식을 보면 v gradient가 w와 orthogonal한 방향임을 알 수 있다. 이 때 w는 v와 proportional하기 때문에, v는 그 자신의 gradient와 orthogonal하다. 즉 업데이트된 $$v'$$의 크기는 피타고라스 정리에 의해 $$\mid\mid v'\mid\mid^2 = \sqrt{1+C^2}\mid\mid v\mid\mid \geq \mid\mid v\mid\mid$$으로 non-decreasing한다. 따라서 large-step에 의해 업데이트된 $$v'$$는 norm의 크기 역시 큰 폭으로 증가하기 때문에 자체적으로 learning-rate를 stabilize하는 역할을 한다. 어느 정도 update step을 크게 겪고 난 이후에는 weight space가 잘 바뀌지 않기 때문에 사용 가능한 learning rate의 범위가 더 넓어진다. 이는 이어지는 layer normalization 논문에서도 비슷하게 지적하는 내용이다.   
  
**1.3 Layer Normalization**  
Weight Normalization이 weight space를 normalize하는 반면, layer normalization은 output activation을 normalize한다. 
물론 그 과정에서 weight을 implicit하게 고려하게 된다. 
RNN을 예로 들면 다음과 같다.     
![image](https://user-images.githubusercontent.com/46081019/60782482-c76a5b00-a181-11e9-85bc-b91de5dd0e92.png)
  
반복해서 강조하지만 normalization의 목표 중 하나는 gradient의 안정화이다. 따라서 loss의 "curvature"를 살펴 보는 것이 좋다. Riemannian manifold에서의 curvature는 riemannian metric 혹은 $$ds^2$$으로 확인할 수 있는데, 이는 parameter space을 따라 model의 output이 얼마나 급격하게 바뀌는지를 의미한다. 말이 어렵지만 단순하게 보면 결국 $$D_{KL}[P(y \mid x; \theta) \mid\mid P(y \mid x; \theta+\delta)]$$를 의미하며 이는 앞 절에서 살펴 보았듯이 fisher information matrix F로 근사할 수 있다. ($$\frac{1}{2}\delta^{T}F(\theta)\delta$$)   
  
![image](https://user-images.githubusercontent.com/46081019/60782969-883d0980-a183-11e9-9abd-434e9bc7f0ad.png)  
결국 parameter space를 geometric한 관점에서 봤을 때 fisher matrix는 model의 outcome(y)이 parameter 변화(delta)에 얼마나 민감한지를 의미하는 metric이다. 논문에서는 GLM을 기반으로 한 실험을 통해 normalization이 fisher information matrix에 어떤 영향을 미치는 지 밝혔다. 
Normalized되지 않은 GLM에서 fisher matrix은 incoming weight, data의 scale에 큰 영향을 받지만, layer normalized GLM은 정규화를 거치기 때문에 fisher matrix가 정규화된 prediction error에 영향을 받는다.   
  
![image](https://user-images.githubusercontent.com/46081019/60783267-d7376e80-a184-11e9-84e2-824582d7a4f6.png)  

따라서 input과 parameter의 scale에 큰 영향을 받지 않기 때문에 더 안정적이고 좋은 gradient를 얻게 된다. 또한 이는 weight space의 scale에 따른 learning rate reduction 효과도 얻을 수 있는데, fisher matrix가 normalization scalar $$\sigma$$에 반비례하기 때문에 weight norm이 커질 경우 fisher matrix 값이 줄어들게 된다. 이는 곧 weight space 방향의 curvature가 원만해지는 것을 의미하며, 따라서 weight vector의 norm이 굉장히 클 경우 gradient update를 통해 weight의 방향을 확 바꾸는 것이 어려워진다. 이를 통해 학습이 진행됨에 따라 weight training이 자동으로 "early-stop"되는 효과를 기대할 수 있다.   
한편 앞 절에서 살펴본 weight normalization은 data의 정규화 과정이 없기 때문에, fisher information matrix가 input data scale에 영향을 받게 된다. 개인적으로는 이 점 때문에 weight normalization보다 layer normalization에 더 신뢰가 간다. Weight normalization의 역할을 layer normalization이 똑같이 혹은 더 잘 해줄 수 있다고 생각한다. 
![image](https://user-images.githubusercontent.com/46081019/60783320-0c43c100-a185-11e9-8a7e-d071873619b8.png)  
  
  
**2. How Batch Normalization Help Optimization?**  
지금까지 흐름을 정리하면, normalization의 목표는 결국 '급격한 gradient update를 피하자'라고 할 수 있겠다. 보다 smooth한, 보다 flat한, 보다 stable한 loss space와 gradient를 얻어야 학습의 수렴성과 일반성을 기대할 수 있을 것이다. 이를 위해 Hessian of negative log-likelihood의 근사라고 할 수 있는 fisher information과 layer normalization까지 알아 보았다.  
  
비슷한 관점에서 batch normalization의 효과를 해석하는 논문이 2018년도 NIPS에 나왔다. Batch normalization이 잘 되는 이유는 loss 및 gradient의 landscape을 부드럽게 만들기 때문이라는 것이다. 이를 통해 gradient는 보다 stabilize되고 predictable하며, 따라서 보다 큰 learning rate을 사용할 수 있고 수렴 속도가 빨라진다. 이런 특징은 앞서 언급한 다른 normalization 기법들과 비슷하다. 
  
논문이 이에 앞서 우선 강조한 내용은 batch-norm paper의 주장과 달리 batch-normalization은 internal covariate shift을 해결하지 않으며, 그럼에도 성능은 향상된다는 것이다.  
![image](https://user-images.githubusercontent.com/46081019/61360602-ba541700-a8b9-11e9-84ea-049d25b0bad4.png)  
위 실험은 hidden layer의 중간에 time-varying whiten noise를 추가했는데, batch normalization이 있는 분홍색 네트워크가 noise로 인한 artificial internal covariate shift를 해결하지 못하고 있다. 그럼에도 성능은 기존 batch normalization과 비슷하다.  
  
![image](https://user-images.githubusercontent.com/46081019/61360760-0a32de00-a8ba-11e9-988a-e16d6624de2d.png)  
더 나아가 위 실험에서는 internal covariate shift를 직접 formulate하여 정량화했다. ICS의 식으로 저자들이 제안한 방식은 어떤 $$ith$$ layer를 기준으로, previous layer들이 gradient에 의해 업데이트된 다음 계산한 gradient와, previous layer가 그대로 유지됬을 때 계산한 gradient의 차이이다. 식으로는 다음과 같다.  
$$G_{t, i} = \triangledown_{W_i^{(t)}} L(W_1^{(t)}, ..., W_k^{(t)}; x^{(t)}, y^{(t)})$$  
$$G'_{t, i} = \triangledown_{W_i^{(t)}} L(W_1^{(t+1)}, ..., W_{i-1}^{(t+1)}, W_{i}^{(t)}, W_{i+1}^{(t)},W_k^{(t)}; x^{(t)}, y^{(t)})$$  
$$ ICS = \parallel G'_{t, i} - G_{t, i} \parallel_2 $$  
이 때 그래프를 보면 Batch Normalization을 추가했을 때 성능은 좋아졌지만, ICS의 polar coordinate value는 더 unstable하다.   
  
논문은 이에 이어서 batch normalization이 잘 되는 이론적 근거로 BN이 loss의 Lipschitzness을 낮추는 것을 근거로 들고 있다. Lipschitzness에 대한 위키피디아의 설명이 굉장히 잘 되어 있다;  
>Intuitively, **a Lipschitz continuous function is limited in how fast it can change: there exists a real number such that, for every pair of points on the graph of this function, the absolute value of the slope of the line connecting them is not greater than this real number;** the smallest such bound is called the Lipschitz constant of the function (or modulus of uniform continuity).   
  
수식으로 연속 함수 f의 Lipschitz condition L을 표현하면 다음과 같다;  
$$ \mid f(t, u) - f(t, v) \mid \leq L \mid u - v \mid $$
그리고 필수 조건은 아니지만, 이를 다음 식으로도 판정할 수 있다;  
*If a partial derivative of f is bounded s.t.*  
$$\mid \frac{df}{dy}(x, y) \mid \leq K,    \forall (x, y) \in D, Then Lip(f)=K$$  
증명은 임의의 닫힌 구간 [u, v]를 정의하여 주어진 식을 적분하고, 부등호 관계를 적용하면 위의 lipschitz condition과 동일하게 만들 수 있다.   
  
우리 네트워크를 non-linear function space의 한 점 f로 보고, f의 loss와 gradient에 대해서 Lipschitzness (혹은 Beta-smoothness)를 낮춘다고 하자. 이는 곧 loss function이 보다 부드럽고, 따라서 gradient가 reliable해짐을 의미한다. 다시 말해 batch-normalization이 없을 때의 loss function에 비해 sharp minima를 줄일 수 있고, exploding or vanishing gradient의 문제도 완화할 수 있다. 이로 인해 gradient가 predictable하면, 다시 말해 방향이 보장되기 때문에, step size를 크게 가져갈 수 있고 이는 학습 속도의 향상으로 연결된다.  
  
  
Batch Normalization은, Batch Normalization이 적용된 후의 gradient norm이 normalization전의 gradient norm보다 매우 높은 확률로 lower bound임을 보임으로써 이를 증명한다.   
![image](https://user-images.githubusercontent.com/46081019/61368093-1756c980-a8c8-11e9-8669-07e2e2f46a9d.png)  
$$\gamma / \sigma$$의 scale로 gradient norm이 줄어드는데, 이는 경험적으로 empirical variance $$\sigma^2$$가 크기 때문에 lipschitz constant의 flatness에 기여할 수 있다.  
  
*Concept of Proof)*  
Batch Normalization의 gradient 유도를 먼저 
