
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
  - Linear convolution으로 good abstraction을 얻기 위해서는 많은 kernel이 필요하다.  
    - 이는 다음 레이어가 고려해야 할 input kernel이 그만큼 많아져버리는 것을 의미한다.  
    - 따라서 more higher level concpet을 얻기 전의 각 local layer에서 좋은 abstraction을 얻고 넘어가는게 좋다.  

3. Network in Network
  ![image](https://user-images.githubusercontent.com/46081019/50895744-4b998b00-144a-11e9-90cf-f8956e78ddbe.png)  
  - 위 수식이 Mlpconv layer이다.
    - k1, k2, ..., kn kernels에 대해 각 layer의 input feature map이 MLP를 통과하고, 그 다음 ReLU를 통과한다.
    - 이를 cross channel parameteric pooling이라 한다. (각 input features-channel-이 crossed되므로)
    - 이는 수식적으로 1x1 convolution kernel과 동일하다.
    - 이 cross channel-pooled layers를 모아서 네트워크 가장 뒷단에서 평균내는 것을 Global Average pooling이라 한다.
      - 논문에서는 이를 feed-forward -> softmax의 대체재로 제안한다.
      - Good Interpretability, No overfitting, Spatial translation available
  - maxout layer와의 차이점
    - 본질적으로 maxout layers는 linear하다.
