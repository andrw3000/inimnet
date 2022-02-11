# Imbedding Deep Neural Networks 
Repository with sample code to support the paper

Andrew Corbett, Dmitry Kangin, Imbedding Deep Neural Networks, ICLR 2022

[[OpenReview](https://openreview.net/forum?id=yKIAXjkJc2F)][[arXiv](https://arxiv.org/abs/2202.00113)]


We include a selection of notebooks to support the results in the paper. These are implemented in Tensorflow using Google Colab. We also include a PyTorch implementation, [inimsolve](inimsolve), in development.


# Experiments
The experiments are contained within the 'notebooks' folder and are adapted for Google Colab. 

The rotating MNIST and bouncing balls data have been obtained from [Yildiz et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/99a401435dcb65c4008d3ad22c8cdad0-Abstract.html) 
via the [implementation](https://github.com/uncbiag/neuro_shooting/tree/master/demos/neurips20) of [Vialard et al., NeurIPS 2020](https://github.com/uncbiag/neuro_shooting/tree/master/demos/neurips20)

## Rotating MNIST
The experiment in section 4.1.1 of the paper compares the proposed model's performance with the existing ones. 
| p-layer | Prediction | 
| :---  | :--- |
| GT    | <img src="figs/data_sample_gt_testing_230.png" width="600" > |
| p=0   | <img src="figs/seed_7_data_sample_pred_training_230_0.png" width="600" > | 
| p=1   | <img src="figs/seed_7_data_sample_pred_training_230_1.png" width="600" > |
| p=2   | <img src="figs/seed_7_data_sample_pred_training_230_2.png" width="600" > | 
| p=3   | <img src="figs/seed_7_data_sample_pred_training_230_3.png" width="600" > | 
| p=4   | <img src="figs/seed_7_data_sample_pred_training_230_4.png" width="600" > | 

The experiment was performed using [notebooks/run_rotmnist.ipynb](notebooks/run_rotmnist.ipynb)

## Bouncing balls
The bouncing balls experiment in section 4.1.2 shows the performance of the method on the bouncing balls task. 
| p-layer | Prediction | 
| :---  | :--- |
| GT    | <img src="figs/bb_gt.png" width="600" > |
| p=0   | <img src="figs/bb_l0.png" width="600" > | 
| p=1   | <img src="figs/bb_l1.png" width="600" > |
| p=2   | <img src="figs/bb_l2.png" width="600" > | 
| p=3   | <img src="figs/bb_l3.png" width="600" > | 


The link on the notebook to reproduce the experiments: [notebooks/discrete_InImNet_tensorflow_time_series_bouncing_balls.ipynb](notebooks/discrete_InImNet_tensorflow_time_series_bouncing_balls.ipynb)


---

Citation details:
```
@article{corbett2022inimnet,
  title={Imbedding Deep Neural Networks},
  author={Corbett, A. and Kangin, D.},
  journal={ICLR},
  fjournal={International Conference on Learning Representations}
  year={2022}
}
```
