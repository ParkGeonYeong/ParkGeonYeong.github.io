---
titel: "Information Theory"
use_math: true
---

**0. 정보란 무엇인가?**  
정보이론에서 말하는 정보란 degree of surprise를 의미합니다. 

**1. 엔트로피란 무엇인가?**
엔트로피는 전달될 수 있는 정보의 기댓값입니다. 우선 정보의 의미를 확실히 정의합시다.
어떤 random variable $$x$$가 확률적으로 매우 낮은 state에 도달했을때 많은 정보를 얻을 수 있습니다. 따라서 어떤 random variable의 특정 state $$x_i$$가 발생할 확률을 $$p(x_i)$$라 했을때 정보량은 이와 반비례함을 알 수 있습니다. 또, 일련의 독립적인 random variable $$x, y$$가 각각 다른 state에 도달했을때 정보의 총량은, 각 $$x$$, $$y$$에서 발생하는 정보의 합입니다. 이때 독립 사건의 확률 $$p(x,y) = p(x)p(y)$$이기 때문에 정보량은 확률의 product 형태를 받아 summation 형태를 return해야 하는 것을 알 수 있습니다. 이를 통해 정보량은 확률의 logarithm 형태라고 정할 수 있습니다.  
종합하면 state $$x_i$$에서 정보량 $$h(x)$$는,  
$$h(x_i) = -log_2p(x_i)$$임을 알 수 있습니다. (이때 base=2는 정보를 bit로 표현하기 위해 임의로 설정한 값입니다)  

이제 x의 모든 state에 대해 이 정보량의 기댓값을 계산하면,  
$$-\sum_{Allx_i}^{}p(x_i)log_2p(x_i)$$임을 알 수 있습니다.  
가장 처음 말했듯이 이 term을 엔트로피 $$H(x)$$라고 부릅니다.  
  
정성적으로 접근한 방식 같지만 사실 이 엔트로피는 우리가 열역학적으로 정의하는 그것과 유사하게 유도될 수 있습니다. 열역학에서 자주 등장하는 예제인 N objects를 bin에 나눠 넣는 경우의 수를 생각해봅시다. $$n_i$$를 ith bin의 element 수라고 하면 전체 가능한 object 나열 경우의 수는  
$$W = \frac{N!}{\prod_{i}^{}n_i!}$$입니다. 이때 분모는 각 bin에서 $$n_i$$ element가 내부적으로 재배열 되는 경우를 고려하지 않기 위한 값입니다.  
이때 엔트로피는 $$H = \frac{lnW}{N}$$으로 정의됩니다. 이때 자주 등장하는 근사식인 $$lnN!\simeq NlnN-N$$을 이용해 식을 정리하면,  
$$\frac{1}{N}(lnN!-\sum_i^{}lnn_i!) = \frac{1}{N}(NlnN-N-\sum_i^{}(n_ilnn_i-n_i)) = \frac{1}{N}(\sum_i^{}(n_ilnn_i-n_ilnN))$$이고, 이는 곧  
$$-\sum_i^{}(\frac{n_i}{N}ln\frac{n_i}{N}) = -\sum_i^{}p_ilnp_i$$가 됩니다. 즉 정보 엔트로피는 얼마나 state가 무질서하게 퍼져있는지를 의미합니다. 엔트로피는 항상 증가하기 때문에, $$max H(x)$$일 때의 $$x$$는 가장 무질서한, uniform distribution을 띕니다. 물론 x가 Continuous한 경우에도 pdf를 이용해 비슷한 방식으로 엔트로피를 구할 수 있습니다.

**2. 이걸 왜 배우는가?**
