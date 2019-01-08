
## Paper Review
# Network in Network

Min Lin, et al, 2013  

핵심 : Network 안의 micro neural network가 receptive field의 input을 abstract한다.  
Global average pooling  
장점 : Interpretability, Robust against to overfitting  
<https://arxiv.org/pdf/1312.4400v3.pdf>  


1. Introduction  
- Generalized Linear Model (GLM) 대비 non-linear function approximator가 얼마나 더 좋은지 강조  
  - GLM은 Linearly separable한 문제에만 적절  
  - CNN 역시 본질적으로 GLM *(사실 non-linear activator와 결합될 시 non-linearity를 가질 수 있다.)*  
  - 이를 극복하고자 non-linear Mlpconv layer 도입(multilayer perceptron의 약어)  
    - 이것이 결국 1x1 Conv layer가 된다.  
    ![default](https://user-images.githubusercontent.com/46081019/50833253-f1d68980-1393-11e9-82be-cb4a1674b2d4.PNG)
    
- Mlpconv의 작동  
  - 하나의 network를 sliding  
  - 각 Convolution kernel 집합의 global average(일종의 combination) pooling 역할을 수행  
  - **이는 각 kernel의 contribution을 확인할 수 있어 interpretability가 높다.**  
  - **또한 단순 fully connected layer 대비 overfitting에 강하다**  
  
2. CNN (의 문제점)
  - 
  
