---
title: "Attend to Attention; What is attention?"
use_math: true
classes: wide
layout: single
---

**본 자료는 다음을 주로 참고했습니다.**  
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/abs/1601.06733)
- [Show, attend and Tell](https://arxiv.org/abs/1502.03044)
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Karl Friston-Free energy principle](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf)
- [Lil'log blog post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [Google AI blog post-Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
  
  
'Attention is a Process such that deciding extent of information flow of each keys to object'이라고 생각한다. 
자연 지능 개체들은 보상에 직간접적으로 관련된 자극을 수용할 때, 기존에 갖고 있던 예측과 어긋날 경우 이 예측 오류(prediction error)를 최소화하기 위해 
attention을 사용한다. 과거의 어떤 key가 현재 인지한 object와 연관이 있는지 파악하는 것이다. 
이러한 attention 개념이 deep neural network에도 도입 되면서 image, video, reinforcement learning, audio 등 sequential data를 사용하는 
다양한 분야에 활용되고 있다. 여기서는 특히 attention이 활발히 사용되는 NLP, neural translation 분야 논문을 구조에 초점을 맞춰 살펴본다. 
다음 포스트에서 reinforcement learning, meta-learning 등 타 분야에서의 attention 활용을 알아본다. 
  
  
**0. Neural Machine Translation by Jointly Learning to Align and Translate**  
- New keywords introduced : Attention, Additive(MLP) attention
- 기존 translation model인 seq2seq의 fixed-length-context vector 문제를 해결
  - Seq2seq는 encoder(using source)의 output인 context vector를 고정시키고(hidden states), 이를 이용해 별도의 decoder에서 target sequence를 생성
  - 이 과정에서 context vector는 encoder의 가장 마지막 hidden vector이며 따라서 길이가 고정되어 있다.
  - 이는 soource sequence가 지나치게 길 경우 제대로 input information을 전달하기 어렵다는 단점 존재
  - Long-term dependency problem
- Target word와 input source sequence를 'align'시킴으로써 이를 해결
- 모델 구조
- ![image](https://user-images.githubusercontent.com/46081019/57978874-10d6dd80-7a50-11e9-89b6-fcae7be9747f.png)  
- Let $$p(y_i \mid y_1, ..., y_{i-1}, x) = g(y_{i-1}, s_i, c_i)$$
  - y는 target sequence, $$s_i$$는 i번째 RNN hidden state
  - $$c_i$$는 i번째 'aligned context vector'로써, $$c_i = \sum_{j=1}^{length of X}{\alpha_{ij}h_j}$$
  - 이때 $$h_i$$는 i번째 annotation으로, bi-directional RNN encoder의 i번째 hidden state를 의미함
    - $$x_i$$에 대한 local한 정보를 주로 갖고 있는 state
  - $$\alpha_{ij}$$는 해당 annotation이 $$y_i$$를 위한 정보를 얼마나 갖고 있는지에 대한 importance weight라고 생각할 수 있음
    - 이 weight가 곧 어떤 input token에 attention을 크게 줄지 결정
  - 이 weight는 다음과 같이 계산됨
  - ![image](https://user-images.githubusercontent.com/46081019/57978986-dec67b00-7a51-11e9-94fd-b964cd5642b3.png)  
    - $$a$$는 two-layered-tanh-MLP이다. $$a$$의 output이 softmax를 거쳐 확률로 변환되고, 이는 각 $$h_j$$에 곱해진다.
    - **즉 attention weight는 target decoder의 $$i-1$$번째 hidden state와, encoder의 모든 $$h_j$$ hidden state를 MLP에 넣어 어떤 $$h_j$$가 
    현재 우리의 관심사인 i-th decoder output token or hidden state와 연관이 높은지 학습한다.**
  - 참고로 decoder의 hidden state $$s_i$$는 다음과 같이 학습된다.
  - ![image](https://user-images.githubusercontent.com/46081019/57979082-80020100-7a53-11e9-99e8-bfcba6e14d92.png)  
    - 모든 hidden state은 n-dimensional이며, 새롭게 계산된 context vector $$c_i$$는 (2n,) dimension이다. (Bi-directional RNN이기 때문)
    - 따라서 C matrix (n, 2n)으로 context vector를 projection시켜 hidden state의 새로운 갱신값에 사용한다.
  - 결과
  - ![image](https://user-images.githubusercontent.com/46081019/57979116-dd964d80-7a53-11e9-9680-b5c0e102f00f.png)  
    - Attention weight 값을 알기 때문에 각 input과 target의 alignment를 visualization할 수 있다.
  
  
**1. Long Short-Term Memory-Networks for Machine Reading**  
- New keywords introduced : Self-attention(Intra-attention), LSTMN
- "How to render sequence-level networks better at 'handling structured input'"
- Self-attention과 Long Short-Term Memory-Networks을 활용해 Source token의 새로운 representation을 고안
  - 현재 token을 이전 tokens의 정보로 표현하여 machine reading, abstractive summarization 분야에 많이 활용됨
  - 'Attention is all you need' 논문에서도 self-attention을 이용해 input sequence를 abstract하면서 시작함
- ![image](https://user-images.githubusercontent.com/46081019/57979166-affdd400-7a54-11e9-891a-44d5de9c2afe.png)  
- 이는 인간의 읽기 과정과도 비슷한데, 문장을 word-by-word로 읽어 가면서 현재 발화에 대한 정보 및 의미를 이전 발화의 묶음에서부터 추출함
- Sequence-level network 문제 제기
  - Vanishing and Exploding gradient
  - Input sequence should be compressed into a single vector
  - *Input의 구조적 특징을 학습, 확인할 수 없음*
    - 한 번에 뭉개서 context vector를 만드는 경우, 언어 및 문장 내 inherent structure 등을 파악할 수 없음
    - 이 점이 self-attention을 통해 해결할 수 있는 가장 큰 장점
- 이 점에서 단순한 LSTM, RNN은 self-attention에 적합하지 않다. 
네트워크 내부적으로 Memory에 input이 재귀적으로 더해지면서 'compression'이 일어나고, 시퀀스를 정확하게 기억하기 어려워진다.
- 따라서 RNN에 별도의 memory와 attention을 도입함으로써 implicit relation between tokens을 학습
  - Long Short-Term Memory-Networks
  - LSTM의 Next state $$h_{t+1}$$는 항상 current state $$h_t$$을 통해 만들어진다
  - **이는 곧 hidden state 간의 Markov property를 가정하는 것이며, $$f(h_{t+1} \mid h_1, h_2, ..., h_{t-1}, h_t) = f(h_{t+1} \mid h_t)$$
  - 그러나 실제로는 LSTM이 bounded memory를 갖고 있기 때문에, sequence가 길거나 메모리가 작은 경우 예외 발생
  - 따라서 모든 token에 대한 information을 aggregate하지 않고, token 간의 관계를 explicit하게 학습하는 과정이 없다
- LSTMN : Memory-Network, LSTM의 memory cell을 network로 span하여 기존의 token을 explicit한 input으로 받음
  - $$h_t$$가 non-markovian manner로 표현 가능하며, memory를 read할 때 attention을 사용할 수 있다.
  - ![image](https://user-images.githubusercontent.com/46081019/57979464-e89fac80-7a58-11e9-9415-958a07d83f11.png)  
- Current input $$x_t$$와 $$x_1, ..., x_{t-1}$$의 관계를 $$h_1, ..., h_{t-1}$$로 표현
  - LSTMN 구조 (자세한 notation은 original paper 참고)
  - ![LSTMN](https://user-images.githubusercontent.com/46081019/57979977-96ae5500-7a5f-11e9-9c7d-cbba57f7b252.png)  
  - Hidden/Memory state 'tape'이 previous token과의 relationship을 저장하고 있다. 
- 모델 구조
  - ![image](https://user-images.githubusercontent.com/46081019/57979993-f1e04780-7a5f-11e9-9554-4ef31346d6bb.png)  
    - Shallow attention : LSTMN이 사용된 점을 제외하면 일반적인 additive attention과 동일하다. (Intra attention=self attention)
    - Deep attention : Additive attention에서 $$h_t, c_t$$가 아닌 hidden, memory 'tape' $$h'_t, c'_t$$를 사용
      - 따라서 encoder의 hidden states group과 decoder의 group 사이 Deep한 관계를 표현 가능
- 결과
- ![image](https://user-images.githubusercontent.com/46081019/57980044-b42fee80-7a60-11e9-8e97-e880f41ede9a.png)
  - Token 간의 valid lexical relations을 인코딩할 수 있음  
  
  
**2. Show, attend and Tell**  
- New keywords introduced : Image-Captioning with attention
- 기존 image-captioning 문제를 attention 도입을 통해 해결
  - 이를 통해 모델이 sailent object에 gaze하는 것을 visualization할 수 있음
  - 이 과정은 Hard attention / Soft attention으로 나뉜다.
    - Hard attention : Feature map의 특정 부분만을 context vector로 제공
    - Soft attention : Weight assigning
- 전체 구조
- ![image](https://user-images.githubusercontent.com/46081019/57980597-5c958100-7a68-11e9-8a3a-9fc42ccd145b.png)  
- 문제 세팅
  - **Encoder : Use previous caption token AND visual context information from convolutional feature map**
    - Let sequence of caption token $$y={y_1, ..., y_C}, y_i \in R^k$$
      - $$k$$ : size of vocabulary
      - $$C$$ : Caption 길이
    - Encoder takes another input as annotation vectors; feature vector extracted from convolutional map
      - $$a={a_1, ..., a_L}, a_i \in R^D$$
      - L : Feature-map-wise vector의 개수 (논문에서는 14x14)
      - D : Feature map의 개수 (논문에서는 512)
  - Decoder : Use LSTM
    - ![image](https://user-images.githubusercontent.com/46081019/57980760-1ccf9900-7a6a-11e9-8c84-e1d56cd7424b.png)  
    - $$T_{D+m+n, n}$$ : (Context+embedding+hidden dimension, hidden dimension)
    - $$\hat{z}_t = \phi(\left\{a_i\right\}, \left\{\alpha_i\right\})$$
      - Context vector
    - Hard attention
      - Strictly assign choice probability of specific i-th location out of L vectors as 0 or 1
      - $$p(s_{t,i}=1 \mid s_{j<t}, a)=\alpha_{t,i}$$
      - $$\hat{z}_t = \sum{s_{t,i}*a_i}$$
      - 이 때 $$s_{t,i}$$ sampling distribution은 multinoulli distribution을 따름
      - By jensen's inequality, 
        - ![image](https://user-images.githubusercontent.com/46081019/57981193-974ee780-7a6f-11e9-92cb-e279973a75b0.png)  
        - Our loss's lower bound is denoted by $$L_s$$
      - 따라서 lower bound을 tighten하기 위해 gradient optimization을 사용해야 한다.
      - 그러나 현재 $$s_{t,i}$$를 랜덤하게 샘플링하고 있기 때문에 미분이 불가능한 loss 식이다.
        - VAE의 reparameterization trick과 비슷
        - [VAE and reparameterization 포스트 참고](https://parkgeonyeong.github.io/Variational-Auto-Encoder(VAE)/)
      - 이를 극복하기 위해 monte-carlo sampling을 거쳐 gradient를 근사한다.
        - ![image](https://user-images.githubusercontent.com/46081019/57981223-0593aa00-7a70-11e9-9989-6d5410325fbb.png)  
        - N개의 feature vector을 sampling 했음
      - 이 때 sampling으로 인해서 log-likelihood $$logp(y \mid s^n, a)$$가 크게 bias될 수 있으므로, 이동 평균 기법을 사용해 variance를 줄인다.
        - $$b_k = 0.9*b_{k-1}+0.1*logp(y \mid s^n, a)$$
        - 이를 log-likelihood의 baseline으로 사용해서, baseline보다 높아지는 경우 positive하게 update해준다.
        - 또한 s 분포의 entropy를 더해서 regularization한다.
      - 결국 최종 update식은 다음과 같다.
        - ![image](https://user-images.githubusercontent.com/46081019/57981262-881c6980-7a70-11e9-9b7c-60a7667cc79a.png)  
      - **Policy를 $$s^n$$의 선택이라고 생각해 보자. 이때 식이 (current likelihood-baseline)을 reward로 하고, policy entropy가 더해진, policy gradient와 동일함을 알 수 있다.**
        - [Policy gradient algorithm 포스트 참고](https://parkgeonyeong.github.io/Policy-Gradient,-Actor-Critic-and-A3C/)
      - 따라서 REINFORCE(Williams, 1992) 알고리즘을 사용해 learning이 가능하다.
    - Soft attention
      - 사실 길게 설명했지만 hard attention은 잘 안쓴다.
        - Learning도 복잡하고, vector 선택을 binary하게 강제하기 때문에 성능이 좋지 않으며 시각화에도 안 좋다
      - Soft attention은 앞서 0. Neural Machine Translation by Jointly Learning to Align and Translate에서 사용한 additive attention model을 사용한다.
      - Doubly Stochastic Attention : i-th token에 대한 attention weight는 당연히 전체 vector에 대해서 합이 1이여야 한다. 
      그런데 이때 모델의 attention이 너무 특정 픽셀군에만 쏠리면 제대로 정규화가 안 되었다고 볼 수 있다. 
      따라서 이에 대한 regularization 효과로 $$\sum{\alpha_{ti}}=1$$을 Loss에 추가해준다. 
  - 결과
    - ![image](https://user-images.githubusercontent.com/46081019/57981366-7091b080-7a71-11e9-9526-a1fdaf3af02f.png)


**3. Attention is all you need**  
- Attention without RNN and CNN
- 
