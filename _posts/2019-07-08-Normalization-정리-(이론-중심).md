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
이때 naive normalization layer $$l_N$$을 추가하면, 이전 layer의 bias update를 
