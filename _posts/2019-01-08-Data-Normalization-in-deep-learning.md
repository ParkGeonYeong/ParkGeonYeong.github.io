---
title: "Data Normalization in Deep learning"
layout: single
classes: wide
---

**1. Data Normalization in Deep Learning**

결론 : Weight Initialization 외에 input data initialization도 중요하다.  
출처 : <http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html>  

- 용어 정리
  - Rescale : 특정 상수를 더하고/빼고/곱하고/나누는 변환. Unit conversion에 자주 활용    
  - Normalize : Minimum과 range를 활용, 0-1사이에 맞추기   
  - Standardize : mean과 std를 활용   
  
- INPUT NORMALIZATION
  - distance function을 activation으로 활용하는 NN (RBF) 등은 매우 중요!  
    - Input range가 0-1인 데이터와 0-10000인 데이터를 함께 training시킬 경우 0-1 데이터의 variance contribution은 무시된다.  
    - kNN, SVM 등에서 데이터를 normalize하는 것과 비슷한 원리  
  - MLP 등의 NN에서는 이론적으로는 weight and bias adjusting으로 이러한 점을 *이론적으로는* 상쇄시킬 수 있을 것이지만, practical하지 않음  
    - 각 hidden unit의 weight를 orientation으로 하는 hyperplanes가 있다.  
    - hyperplane의 bias가 존재한다.   
    - 이 hyperplane들이 주어진 데이터를 분류해내기 위해서는 data cloud을 대부분 통과해야 한다.  
    - data initialization이 제대로 되지 않고 origin을 크게 벗어나는 경우 hyperplane이 entire data를 구분해내기 어렵다.  
    - 따라서 zero mean의 centered small random variable이 가장 유리하다.   
