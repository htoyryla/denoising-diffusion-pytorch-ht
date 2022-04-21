## Experimenting with image making using denoising diffusion

Based on the excellent and clear codebase by @lucidrains

This is an experimental project with emphasis on visual work. I would not mind if somebody trained a really cool model,
but in general it is best to try in small scale. Even 200 images and a few hours of training.  See discussions.

Model architectures unetcn0 (Unet with convnext) and unet0 (unet with resnet) are known to work. Both copied from @lucidrains repo.

My intention is

* add scripts for convenient training and image generation / transformation
* experiment with image transformation using diffusion
* especially training with small homogenous datasets that can be quickly trained and retrained
* and use the trained models as "brushes" or "filters" 
* starting from seed images and using various means of guidance (text/CLIP, texture/VGG+GRAM etc) 
* experiment with various model structures

Some early results, using datasets of a few hundred images, selected according to visual and not semantic criteria. Using ConvNext based Unet, not the current version but from before the recent changes, training at 512px for 1 - 2 days. It also appears that once we have a pretrained model, it can quite fast be retrained with a visually different set. BTW, I have not succeeded to get the newest Unet/Convnext to train satisfactorily (but it might just be that I have not yet found a correct lr)

<p float="left">
<img src="https://user-images.githubusercontent.com/15064373/163939108-f131d71d-384a-4211-95ab-573e1cfe33ee.jpeg" width="360"></img>
<img src="https://user-images.githubusercontent.com/15064373/163939115-adf16ff6-8ef7-4003-98c0-6cfc604a44c3.jpeg" width="360"></img>
<img src="https://user-images.githubusercontent.com/15064373/163939127-acac4f49-7a42-4c81-928a-5d8d56998a22.png" width="360"></img>
<img src="https://user-images.githubusercontent.com/15064373/163940103-836b4e52-57e5-4375-b292-3d538f1e3f6d.jpeg" width="360"></img>
</p>

## How to train

Using unetcn0 I have been able to train useful models with as few as 250 images, using 512px imagesize and 6+6 layers unet.

```
python diffutrainer.py --images path_to_your_image_folder --lr 5e-5 --steps 1000 --accum 10 --dir output_folder --imageSize 512 --barchSize 2 --saveEvery 100 --nsamples 2 --mults 1 1 2 2 4 8 --model unetcn0

--lr learning rate
--steps diffusion steps
--accum accumulate gradient over x iters
--dir output_folder 
--imageSize native image size for the model to be trained
--batchSize  
--saveEvery save and sample at given intervals
--nsamples how many samples to generate when saving, small number can save much time
--mults multipliers affecting number of feature maps per level, use larger numbers for the later, more abstract levels
        number of multipliers also determines depth of the model, good rule is to  have 5 numbers of 256 and 6 numbers for 512
        too few layers will make model fail to perceive the image as a whole
--model unetcn0 (selects which model architecture is used, to be explained later, this one at least works)
```
## How to sample

Currently supports CLIP guidance and use of seed image. Use lr for tuning how much effect CLIP has. 

```
python diffudiver.py --text prompt_for_clip --dir output_folder --name basename_for_stored_images --image path_to_seed_image --mul 2  --lr 0.0004 --imageSize 1024 --show --modelSize 512 --load path_to_stored_model --mults same_as_in_training --ema --saveEvery 50 --saveAfter 550  --weak 1  --model  unetcn0

--mul --weak affect how seed image is handled
--lr 0.0004 clip guidance lr, use this to find bvalance between text and image
--imageSize 1024 generated image size, multiple of 32
--modelSize 512 native size of the model
--ema  use ema checkpoint (you can test what difference does it make)
--saveEvery 50   saving of frames during iteration
--saveAfter 550  start saving only after given step
--model  unetcn0 as in training
```

## --------------------- Original readme starts here -----------------------------------------

<img src="./denoising-diffusion.png" width="500px"></img>

## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution.

This implementation was transcribed from the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a> and then modified to use <a href="https://arxiv.org/abs/2201.03545">ConvNext</a> blocks instead of Resnets.

<img src="./sample.png" width="500px"><img>

[![PyPI version](https://badge.fury.io/py/denoising-diffusion-pytorch.svg)](https://badge.fury.io/py/denoising-diffusion-pytorch)

## Install

```bash
$ pip install denoising_diffusion_pytorch
```

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Citations

```bibtex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@inproceedings{anonymous2021improved,
    title   = {Improved Denoising Diffusion Probabilistic Models},
    author  = {Anonymous},
    booktitle = {Submitted to International Conference on Learning Representations},
    year    = {2021},
    url     = {https://openreview.net/forum?id=-NEXDKk8gZ},
    note    = {under review}
}
```

```bibtex
@misc{liu2022convnet,
    title   = {A ConvNet for the 2020s},
    author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
    year    = {2022},
    eprint  = {2201.03545},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
