## SAKT-pytorch  
Pytorch Implementation of **"A Self-Attentive model for Knowledge Tracing"** based on https://arxiv.org/abs/1907.06837.    
Given the past history of student in form of questions and answers, model tries to predict the correctness of student's future questions. Previous methods such as DKT, DKVMN etc, were based on RNN, however these model failed to generalize well when sparse data is used. SAKT model identifies the knowledge concepts from the student's past activities that are relevant for prediction of future activities. Self Attention based approach is used to identify relevance between knowledge concepts.  
(Note- This model was trained on [riiid dataset](https://www.kaggle.com/c/riiid-test-answer-prediction) that reached AUC 0.749 on validation set.)
### SAKT model architecture  
  
<img src="https://github.com/arshadshk/SAKT-pytorch/blob/main/from_paper.JPG">
  
## Usage 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import sakt

def randomdata():
    input_in = torch.randint( 0 , 49 ,(64 , 12) )
    return input_in, input_in

d1,d2 = randomdata()


model = sakt( ex_total= 50 , 
              seq_len= 12, 
              dim= 128 ,
              heads= 8 ,
              dout= 0.2)

out = model( d1, d2)


```
## Parameters
- `ex_total`: int.  
Total numbe of unique excercise.
- `seq_len`: int.  
Input sequence length.  
- `dim`: int.  
Dimension of embeddings.
- `heads`: int.  
No. of heads in multi-head attention.    
- `dout`: int.  
Dropout for feed forward layer.    


I would recommend you to have a look at [Tensorflow implementation](https://github.com/shalini1194/SAKT) of SAKT.


## Citations

```bibtex
@article{pandey2019self,
  title={A Self-Attentive model for Knowledge Tracing},
  author={Pandey, Shalini and Karypis, George},
  journal={arXiv preprint arXiv:1907.06837},
  year={2019}
}
```

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```



