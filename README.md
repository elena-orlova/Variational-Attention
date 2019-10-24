# Latent Alignment and Variational Attention

This is an unofficial Pytorch implementation of the paper 'Latent Alignment and Variational Attention'. 

## Requirements

You need ```Pytorch 0.4``` and ```Python 3.6```.

## Running the code 

Firstly, you need to preprocess the data at ```Preprocessing.ipynb```. We use IWSLT 2014 English-German dataset. 

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


\end{itemize}