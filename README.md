# CCN

Code and data for the paper "[Multi-Label Classification Neural Networks with Hard Logical Constraints](https://dl.acm.org/doi/pdf/10.1613/jair.1.12850)"

## Train CCN

In order to train the network use the file ```train.py```. An example on how to run it for the dataset ```emotions```:
```
python train.py --dataset "emotions" --num_classes 6 --seed "$seed" --split "$split" --device "$device" --lr "$lr" --dropout "$dropout" --hidden_dim "$hidden_dim"  --num_layers "$num_layers" --weight_decay "$weight_decay" --non_lin "$non_lin" --batch_size "$batch_size" 
```
```train.py``` saves a pickle file for each execution in the ```hyp``` folder. Each pickle file stores the value of the hyperparameters and of the validation loss. 


## Reference

```
@article{giunchiglia2021,
  author    = {Eleonora Giunchiglia and Thomas Lukasiewicz},
  title     = {Multi-Label Classification Neural Networks with Hard Logical Constraints},
  journal   = {Journal of Artificial Iintelligence Research (JAIR)},
  volume    = {72},
  year      = {2021}
}
```
