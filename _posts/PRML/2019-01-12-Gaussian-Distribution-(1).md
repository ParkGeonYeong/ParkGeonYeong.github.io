---
title : "Gaussian Distribution (1)"
use_math : True
---

**본 포스트는 gaussian distribution을 설명하기 위한 선형대수학적 기초를 먼저 정리하고 있습니다.**  
- Symmetric matrix, Meaning of jacobian에 대해 다룹니다.  
**Gaussian distribution에 대한 내용은 (2)에서 확인하실 수 있습니다.**  
부족함으로 인해 포스트에 오류가 있을 수 있습니다. 

**0. Symmetic matrix**  

Symmetric Matrix는 $$A^\intercal = A$$인 행렬을 의미합니다. Symmetric Matrix는 두 가지 좋은 성질을 가지고 있습니다.
1. Symmetric Matrix A가 실수 행렬인 경우 eigen value는 모두 실수이다.
2. A의 eigen vector는 orthonormal하게 구할 수 있다.
증명은 다음과 같습니다.  

$$Pre-requisite$$  
$$A^\intercal = A$$, $$A^{*} = A$$, *는 conjugate sign, $$Au = {\lambda}u$$에서 $$\lambda$$는 eigen value, $$u$$는 eigen vector  

$$Prof \; 1$$  
$$Au = {\lambda}u$$   
유도의 편의성을 위해, 우변에 $$(u^{*})^{\intercal}$$을 pre-product해보겠습니다.  
$$\lambda((u^{*})^{\intercal})u = (u^{*})^{\intercal}Au = ((u^{*})^{\intercal}A)u$$  
이는 곧 다음과 같습니다;  
$$((u^{*})^{\intercal}A)u = (A^{\intercal}u^{*})^{\intercal}u = (Au^{*})^{\intercal}u\;\; (A\,is\,symmetric)$$  
이때 $$Au^{*} = {\lambda}^{*}u^{*}$$이므로, $$(Au^{*})^{\intercal}u = ({\lambda^{*}}u^{*})^{\intercal}u  
= {\lambda^{*}}((u^{*})^{\intercal}u)$$  
처음과 나중을 비교하면, $${\lambda^{*}}((u^{*})^{\intercal}u) == {\lambda}((u^{*})^{\intercal}u)$$  
이때 eigen vector는 non-zero이기 때문에 conjugate term과의 곱은 non-negative입니다.  
따라서 $${\lambda^{*}} == {\lambda}$$로, eigen value는 실수입니다.  
  
$$Prof \; 2$$  
$$Au_1 = \lambda_1u_1$$, $$Au_2 = \lambda_2u_2$$인 두 eigen value-vector가 있다고 합시다.  
이때 $$\lambda_1u_1$$에 $$u_2^{\intercal}$$를 곱합시다.  
$$\lambda_1u_2^{\intercal}u_1 = u_2^{\intercal}Au_1 = (A^{\intercal}u_2)^{\intercal}u_1$$  
$$(A^{\intercal}u_2)^{\intercal}u_1 =  (Au_2)^{\intercal}u_1 = (\lambda_2u_2)^{\intercal}u_1$$  
$$(\lambda_2u_2)^{\intercal}u_1 = {\lambda_2}u_2^{\intercal}u_1$$  
처음과 나중을 비교하면, $$\lambda_1u_2^{\intercal}u_1 = {\lambda_2}u_2^{\intercal}u_1$$  
이때 서로 다른 eigen value에 대응되는 두 eigen vector의 product는 0이여야 합니다.  
$$(\lambda_1-\lambda_2)u_2^{\intercal}u_1 = 0$$  
따라서 두 eigen vector는 orthogonal하며, 이를 normalize하여 orthonormal vector를 얻을 수 있습니다.  

**1. Jacobian matrix**  
<https://wikidocs.net/4053>을 참고로 작성했습니다.  
내용은 위 링크를 참고하고, Jacobian matrix의 행렬식이 갖는 의미에 집중해 보겠습니다.  
Jacobian matrix는 서로 다른 두 좌표계 $$(x,y), \, (u,v)$$의 transform에서 자주 등장합니다.  
(x,y) 좌표계에서 (u,v) 좌표계로 전환할 때 어떤 그래프의 면적에는 변화가 생깁니다. 우리는 그 차이가 얼마나 발생하는지를 새 coordinate의 미소 간격 du, dv로 표현하고 싶습니다. 결론적으로, jacobian의 행렬식이 그 차이를 대변합니다.  
$$dxdy = det(J(u,v))dudv$$  
이는 곧 3차원(부피 관계)로도 확장 가능합니다. 이를 조금 바꿔 말하면,
$$\int_{\bf x} f({\bf x})d{\bf x} = \int_{\bf y} f({\bf y})|{\bf J}|d{\bf y}$$ 관계가 성립할 수 있다는 뜻입니다 (x,y는 multivariable vector 분포).  
x coordinate를 y coordinate으로 전환할때, 식에서 jacobian의 행렬식은 새 coordinate y에서의 강체의 부피가 이전 대비 얼마나 차이나는지를 보정해주는 값입니다.  
  
(의미적으로 제가 편한 방식으로 해석했기 때문에 많은 오류가 있을 수 있습니다. 언제든지 지적 부탁드립니다.)
