---
title: "Normalization 정리 (이론 중심)"
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
여기서는 normalization에 사용되는 수학적 기반(lipschitzness, fisher information, Hessian 등)을 위주로 지난 벤치마크 논문들을 되돌아 본다. 
  
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
또한 $$z=Fx$$에서 layer $$F$$를 단순히 linear transformation으로 가정하고, normalized x와 z를 각각 uncorrelated gaussian으로 가정할 때 $$I = cov(z) = Fxx^{T}F^{T} = FF^T$$으로, F의 singular value는 1이 되어 gradient의 magnitude를 제한한다. 물론 실제로는 non-linear transformation이지만 BN이 학습을 안정화시킨다는 간접적인 증거이다.  
  
  
**1. Weight Normalization and Layer Normalization**  
Batch Normalization 이후 나온 많은 normalization paper들이 
