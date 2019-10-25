# Latent Alignment and Variational Attention

This is an unofficial Pytorch implementation of the paper [Latent Alignment and Variational Attention](https://arxiv.org/pdf/1807.03756.pdf). 

## Requirements

You need ```Pytorch 0.4``` and ```Python 3.6```.

## Running the code 

Firstly, you need to preprocess the data at ```Preprocessing.ipynb```. We use IWSLT 2014 German-English dataset. 

Then, there are 3 modes are available with training and validation: 

- Soft attention 
```
source va.sh && CUDA_VISIBLE_DEVICES=0 train_soft_mode
 ```

- Variational categorical attention with exact ELBO
```
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_exact_elbo_mode
 ```

- Variational categorical attention with REINFORCE

```
source va.sh && CUDA_VISIBLE_DEVICES=0 train_cat_reinf_mode
 ```

 ## Files description

 - The notebook ```Preprocessing.ipynb``` is for IWSLT 2014 German-English dataset preprocessing
 - A file ```main.py``` is the main script for running the code 
 - A file ```utils.py``` consists of helper functions including functions for training, buiding an optimizer 
 - A file ```va.sh``` consists of arguments for building models/training mode 
 - In a directory ```tools```  there are models and tools for translation 
 
## Additional resourses
- Pipeline regarding preprocessing is avaliable [here](https://github.com/elena-orlova/var-attention-project/blob/master/notebooks/preprocessing.ipynb).
- Several plots description is shown [here](https://github.com/elena-orlova/var-attention-project/blob/master/notebooks/Results%20plots.ipynb)


## Report
Project proposal, formal description and report with results are avaliable [here](https://github.com/elena-orlova/var-attention-project/blob/master/documentation/Proposal_Bayes.pdf).
