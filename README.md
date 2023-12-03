# Jayveersinh Raj
# BS20-DS-01
# j.raj@innopolis.university

The project is Neural Collaborative filtering on movie lens 100k dataset.

# Training
The training notebook can be found under notebooks. The model is trained on 100k samples of movie lens dataset

# Evaluation
## Metrics
**Hit Rate (HR)** : Hit Rate is a binary metric that measures the proportion of correct recommendations in a given recommendation list. It answers the question: "Did the user find at least one relevant item in the top-k recommendations?"

**Normalized Discounted Cumulative Gain (nDCG)**:  NDCG is a ranking metric that evaluates the quality of the entire ranked list of recommendations. It considers both the relevance of the items and their positions in the list.

Formulas for the both can be found under the figures directory of reports. 
Following are the numbers (Mean) over test set of 610 unique users not present in the training.
| HR    | nDCG  | layer for MLP | Epochs |
|-------|-------| --------------| ------ |
| 0.784 | 0.545 |  [64,32,16]   |   10   |


# Example output
<img width="908" alt="image" src="https://github.com/Jayveersinh-Raj/Gujarati_Grammarly/assets/69463767/16f43276-dfcb-40a1-83e8-e91b7d92c1e1">

    
# More details
For further details please refer to the report under the reports directory.

# Reference
The project is based on: [Neural Collaborative filtering](https://arxiv.org/abs/1708.05031)

# Citation

    @article{DBLP:journals/corr/abs-1708-05031,
      author       = {Xiangnan He and
                  Lizi Liao and
                  Hanwang Zhang and
                  Liqiang Nie and
                  Xia Hu and
                  Tat{-}Seng Chua},
      title        = {Neural Collaborative Filtering},
      journal      = {CoRR},
      volume       = {abs/1708.05031},
      year         = {2017},
      url          = {http://arxiv.org/abs/1708.05031},
      eprinttype    = {arXiv},
      eprint       = {1708.05031},
      timestamp    = {Mon, 13 Aug 2018 16:49:05 +0200},
      biburl       = {https://dblp.org/rec/journals/corr/abs-1708-05031.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }


