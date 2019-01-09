---
title: "1.5 Decision Theory"
use_math: true
---

본 글은 PRML 1.5 Decision Theory를 QnA 형식으로 정리했습니다.  
<http://norman3.github.io/prml/docs/chapter01/5>을 참고했습니다.  
의, 오역 및 잘못된 의견이 첨가되었을 수 있습니다.  

**1. Decision Theory가 무엇인가? ** 
- Misclassification rate, 혹은 *Expected loss*를 최소화하는 Decision boundary를 찾는 것입니다.
  - $p(correct) = \sum_{k=1}^{K}\int_{R_k}^{}p(x, C_k)dx  \propto \sum_{k=1}^{K}\int_{R_k}^{}p(C_k|x)p(x)dx$
