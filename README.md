# Image Extension using Partial Convolutions with PyTorch
this repository is adapted from IsHYuhi's implementation of Image Inpainting using Partial Convolution ([link](https://github.com/IsHYuhi/Image_Inpainting_Using_Partial_Convolutions))


## Requirements
listed in requirements.txt

## How to use
* Put images under ```./img```.
Then,
```
python3 train.py
```

## Training
The model was trained using Google Colab, stored in train.ipynb

## References
* Image Inpainting for Irregular Holes Using Partial Convolutions, [Guilin Liu](https://liuguilin1225.github.io/), [Fitsum A. Reda](https://scholar.google.com/citations?user=quZ_qLYAAAAJ&hl=en), [Kevin J. Shih](http://web.engr.illinois.edu/~kjshih2/), [Ting-Chun Wang](https://tcwang0509.github.io/), Andrew Tao, [Bryan Catanzaro](http://ctnzr.io/), **NVIDIA Corporation**, [[arXiv]](https://arxiv.org/abs/1804.07723)
* N. Akimoto, D. Ito and Y. Aoki, "Scenery Image Extension via Inpainting With a Mirrored Input," in IEEE Access, vol. 9, pp. 59286-59300, 2021, doi: 10.1109/ACCESS.2021.3073223.