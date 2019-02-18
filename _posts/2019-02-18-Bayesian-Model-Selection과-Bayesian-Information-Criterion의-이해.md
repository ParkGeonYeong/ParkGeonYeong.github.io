---
title: "Bayesian Model Selection과 Bayesian Information Criterion의 이해"
use_math: true
layout: single
classes: wide
---

본 자료는 다음을 주로 참고했습니다.
- PRML 3.4 Bayesian Model Comparison
- [와튼 스쿨 자료](http://www-stat.wharton.upenn.edu/~stine/research/select.Bayes.pdf)
- [Bayesian Information Criterion 위키](http://www-stat.wharton.upenn.edu/~stine/research/select.Bayes.pdf)

**0.Bayesian Model Selection**  
Bayesian Model Selection은 모델 자체를 parameterize하여 주어진 데이터 상에서 최적의 모델을 선택하는 베이지안 기법이다. 
다양한 선택지 가운데 오버 피팅을 막을 수 있도록 선택을 돕고, 나아가 해당 모델의 multiple and complex parameters를 한번에 정할 수 있다. 
Validation data set을 따로 정해 모델의 성능을 검증할 필요가 없다. (그러나 practical하게는 확인한다)  

베이지안 접근법이 그렇듯 우선 주어진 데이터를 표현함에 있어서 각 모델들의 불확실함을 확률로 나타낸다. 
이때 모델들은 신경망의 경우 resnet, VGGNet 등이 될 수도 있고, 아예 다른 kernel-SVM, Tree 등과 비교할 수도 있다. 
또 Polynomial fitting의 경우 order가 될 수도 있다. 각 모델에 대한 불확실한 정도를 $$p(M_i)$$라 하고 이를 prior probability, 
혹은 모델에 대한 사전 선호도로 사용할 수 있다. 이때 주어진 데이터를 $$D$$라 하고, bayes theorem을 사용하면 다음과 같다.  
$$p(M_iㅣD) \propto p(D|M_i)p(M_i)$$  
이때 likelihood인 $$p(DㅣM_i)$$를 model evidence라고 부르며, 데이터를 통해 확인한 각 모델의 선호도를 의미한다. 
또한 임의의 두 모델의 model evidence 비율 $$p(DㅣM_i)/p(DㅣM_j)$$를 **bayes factor**라고 부른다. 
  
이때 model evidence는 각 모델의 free parameters에 대해 bayesian statistics을 적용해서 구하게 되며 아래 식과 같다.  
$$p(DㅣM_i) = \int{p(Dㅣw, M_i)p(wㅣM_i)}dw$$  
  
  
주어진 model evidence의 의미를 더 풀어 설명하면 다음과 같다. 모델의 파라미터에 대해 사전 분포를 $$p(w)$$으로 잡자. 
또 사후 분포는 매우 sharp하여 maximum likelihood일 때의 파라미터 값에 모여 있다고 하자.
전체 $$w$$ range 중에서 likelihood 관점에서 data를 가장 certain하게 설명하고 있는 w 값을 
'대략' $$w_{MAP}$$이라 하자. 대략 single point estimation이다. 해당 영역의 사후 분포 너비를 $${\triangle}w_{posterior}$$이라 하고, 
전체 prior의 너비를 $${\triangle}w_{prior}$$이라 하자. 그림과 같다.  
![image](https://user-images.githubusercontent.com/46081019/52960481-39612400-33dc-11e9-9378-b8e6bdf00d43.png)  )  
이때 $$p(DㅣM_i)$$은 다음과 같다. $$W_{MAP}$$이외의 likelihood 값은 무시하여 직사각형으로 근사한 결과이다.   
$$p(DㅣM_i) = \int p(D|w,M_i)p(wㅣM_i)dw \simeq p(D|w_{MAP}, M_i)\frac{\Delta w_{posterior}}{\Delta w_{prior}}$$   
로그를 씌우면 다음과 같다.  
$$\ln p(DㅣM_i) \simeq \ln p(D|w_{MAP}, M_i) + \ln{\frac{\Delta w_{posterior}}{\Delta w_{prior}}}$$   
**위 식은 결국 maximum likelihood가 높아야 해당 model evidence, 즉 선호도가 높아지는 것을 의미한다.** 
이때 뒷 항이 패널티 항으로 붙어있는데 이는 $$w_{posterior}$$가 전체 사전 분포 대비 지나치게 sharp한 경우 페널티를 부과하는 항으로써 
오버피팅을 방지하는 일종의 regularizer 역할을 한다고 생각한다. 이때 싱글 파라미터가 아닌 멀티 파라미터가 되는 경우 위의 식은 아래와 같이 바뀌며, 
M은 파라미터의 개수를 의미한다. 지나치게 파라미터가 많고 sharp한 모델은 model evidence가 낮도록 regularize한다고 할 수 있다.  
$$\ln p(DㅣM_i) \simeq \ln p(D|w_{MAP}, M_i) + {M\ln \left( \frac{\Delta w_{posterior}}{\Delta w_{prior}} \right)}$$  
그러나 모델이 complex할 수록 likelihood는 높아지기 때문에 두 항은 서로 trade-off 관계에 있다고 할 수 있다. 이는 다음 예시에서 확인할 수 있다.  
![image](https://user-images.githubusercontent.com/46081019/52962911-cfe41400-33e1-11e9-9a01-86cd47edc184.png)  
위 그림에서 주어진 데이터에 polynomial fitting을 시도할 때 order가 높아질 수록 likelihood는 높아 지지만, 
모델의 선호도는 order=3이 가장 높은 것을 알 수 있다. Order가 너무 낮으면 제대로 데이터에 fitting되지 않고, 
너무 높으면 complexity로 인해 페널티를 받기 때문이다. 즉 Bayesian Model Selection을 통해 overfitting을 피할 수 있는 것을 알 수 있다.  
심플한 모델은 특정 데이터에 확 잘 fitting될 수 있지만 많지 않는 데이터에는 낮은 성능을 보이고, 복잡한 모델은 전체적으로 데이터에 어느정도 fitting되지만 
특정 데이터에 대해서는 그에 걸맞는 간단한 모델과의 경쟁에서 이길 수 없다.  
  
**1. Bayesian Information Criterion**  
앞서 확인 했듯이 모델의 파라미터 $$w$$에 대해 계산을 거쳐 model evidence를 얻을 수 있었다. 
또한 bayes factor를 이용해 두 모델의 evidence을 비교할 수 있었다.
그러나 실제로 이러한 적분 과정을 일일히 거치는 것은 어려움이 있기에 근사(approximate) 과정이 필요하며 
이를 위해 bayesian information criterion (BIC)이라는 메트릭이 만들어지게 된다.  

BIC 역시 likelihood와 동시에, model parameter의 개수에 패널티를 부과하며 
비슷한 metric인 Akaike information criterion (AIC)보다 더 많이 부과한다고 한다. BIC의 수식은 다음과 같다.  
$$BIC = ln(n)k-2ln(L)$$  
이때 n은 number of data points, k는 모수의 개수, L은 max likelihood이다. 
전개 및 증명 과정은 테일러 근사 과정과 laplace's method를 사용하는 것으로 보이는데 정확히 이해하지는 못했다.

