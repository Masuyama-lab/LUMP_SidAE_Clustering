# A Continual Self-Supervised Learning Framework using a Decoder and a Clustering-based Data Selection Method

## Installation
```
$ pip install -r requirement.txt
```

## Run
### ID Condition
E.g. Running LUMP+SidAE+CA+ on CIFAR-100.
```
$ python main_LUMP_clustering.py
```

E.g. Running LUMP+SidAE+CA+ on Tiny-ImageNet.
```
$ python main_LUMP_clustering.py -c configs/sidae_tinyimagenet.yaml
```

The training configurations can be found in the `./configs`.
You can select from the following clustering algorithm:

- `kmeans` (k-means)
- `kmedoids` (k-medoids)
- `sc` (Spectral Clustering)
- `gmm` (Gaussian Mixture Model)
- `som` (Self-Organizing map)
- `caplus` (CA+)

Running LUMP+KM, set `clustering: kmeans` in the configuration file.

The results (accuracy and forgetting) and checkpoints are saved in the `./logs` and `./checkpoints`.

### OOD Condition
After running the ID condition experiment, you can run OOD condition experiments.

E.g.
- Method : LUMP+SidAE+CA+
- Training data : TinyImageNet,
- Test data : MNIST

```
$ python main_LUMP_clustering_OOD.py -c configs/sidae_tinyimagenet.yaml --trained_model_dir ./checkpoints/lump+caplus_sidae_seq-tinyimg/ --ood_data_name seq-mnist 
```
Instead of the command, you can set arguments in `arguments.py`.


## Citation

If you use this code in your research, please cite the following paper:

R. Fujii, N. Masuyama, Y. Nojima, C. K. Loo, and L. W. Shiung, "A continual self-supervised learning framework using a decoder and a clustering-based data selection method," 2026, Accepted.

```bibtex
@article{fujii2026continual,
  author  = {R. Fujii and N. Masuyama and Y. Nojima and C. K. Loo and L. W. Shiung},
  title   = {A Continual Self-Supervised Learning Framework Using a Decoder and a Clustering-Based Data Selection Method},
  year    = {2026},
  note    = {Accepted}
}
```

## Acknowledgement
The code is build upon [divyam3897/UCL](https://github.com/divyam3897/UCL).

