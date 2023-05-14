from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import noise_like

import torch
from torchvision.utils import save_image
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2
from pytorch_msssim import ssim
from postproc import pprocess

from functools import partial

import numpy as np

'''
pip install denoising_diffusion_pytorch
'''

from cutouts import cut

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--text', type=str, default="", help='text prompt')
parser.add_argument('--image', type=str, default="", help='path to init image')
parser.add_argument('--img_prompt', type=str, default="", help='path to image prompt')
parser.add_argument('--tgt_image', type=str, default="", help='path to target image')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--ssimw', type=float, default=1., help='target image weight')
parser.add_argument('--textw', type=float, default=1., help='text weight')
parser.add_argument('--tdecay', type=float, default=1., help='text weight decay')
parser.add_argument('--imgpw', type=float, default=1., help='image prompt weight')
parser.add_argument('--steps', type=int, default=1000, help='diffusion steps')
parser.add_argument('--skip', type=int, default=0, help='skip steps')
parser.add_argument('--dir', type=str, default="out", help='base directory for storing images')
parser.add_argument('--name', type=str, default="test", help='basename for storing images')
parser.add_argument('--mul', type=float, default=1., help='noise divisor when using init image')
parser.add_argument('--show', action="store_true", help='show image in a window')
parser.add_argument('--ema', action="store_true", help='use ema model')
parser.add_argument('--imageSize', type=int, default=512, help='image size')
parser.add_argument('--h', type=int, default=0, help='image height')
parser.add_argument('--w', type=int, default=0, help='image width')
parser.add_argument('--modelSize', type=int, default=512, help='native image size of the model')
parser.add_argument('--saveEvery', type=int, default=0, help='image save frequency')
parser.add_argument('--saveAfter', type=int, default=0, help='save images after step')
parser.add_argument('--low', type=float, default=0.4, help='lower limit for cut scale')
parser.add_argument('--high', type=float, default=1.0, help='higher limit for cut scale')
parser.add_argument('--cutn', type=int, default=24, help='number of cutouts for CLIP')
parser.add_argument('--load', type=str, default="", help='path to pt file')
parser.add_argument('--saveiters', action="store_true", help='')
parser.add_argument('--mults', type=int, nargs='*', default=[1, 1, 2, 2, 4, 8], help='')
parser.add_argument('--weak', type=float, default=1., help='weaken init image')
parser.add_argument('--mid', type=float, default=0, help='weaken init image')
parser.add_argument('--model', type=str, default="", help='model architecture: unet0, unetok5, unet1,unetcn0')
parser.add_argument('--gradv', action="store_true", help='another guidance technique')
parser.add_argument('--showLosses', action="store_true", help='display losses')
parser.add_argument('--spher', action="store_true", help='use spherical loss')
parser.add_argument('--ignore', action="store_true", help='use if steps not same as during training')

parser.add_argument('--sampling_steps', type=int, default=50, help='sampling steps')
parser.add_argument('--eta', type=float, default=0.5, help='ddim eta')


parser.add_argument('--contrast', type=float, default=1, help='contrast, 1 for neutral')
parser.add_argument('--brightness', type=float, default=0, help='brightness, 0 for neutral')
parser.add_argument('--saturation', type=float, default=1, help='saturation, 1 for neutral')
parser.add_argument('--gamma', type=float, default=1, help='gamma, 1 for neutral')
parser.add_argument('--unsharp', type=float, default=0, help='unsharp mask')
parser.add_argument('--eqhist', type=float, default=0., help='histogram eq level')
parser.add_argument('--median', type=int, default=0, help='median blur kernel size, 0 for none')
parser.add_argument('--c1', type=float, default=0., help='do not use')
parser.add_argument('--c2', type=float, default=1., help='do not use')
parser.add_argument('--sharpenlast', action="store_true", help='do not use')
parser.add_argument('--sharpkernel', type=int, default=3, help='sharpening kernel')
parser.add_argument('--ovl0', type=float, default=0, help='blend original with blurred image')
parser.add_argument('--bil', type=int, default=0, help='bilateral filter kernel')
parser.add_argument('--bils1', type=int, default=75, help='bilateral filter sigma for color')
parser.add_argument('--bils2', type=int, default=75, help='bilateral filter sigma for space')




opt = parser.parse_args()

mtype = opt.model

if opt.h == 0:
    opt.h = opt.imageSize

if opt.w == 0:
    opt.w = opt.imageSize
    

if mtype == "unet0":
  from alt_models.Unet0 import Unet
elif mtype == "unet0k5":
  from alt_models.Unet0k5 import Unet
elif mtype == "unet1":
  from alt_models.Unet1 import Unet
elif mtype == "unet2":
  from alt_models.Unet2 import Unet    
elif mtype == "unetcn0":
  from alt_models.UnetCN0 import Unet
else:
  print("Unsupported model: "+mtype)
  exit()

def show_on_screen(image_tensor, window="out", maxsize=720):
    im = image_tensor.detach().numpy()   # convert from pytorch tensor to numpy array
    #print(im.shape)
    
    # pytorch tensors are (C, H, W), rearrange to (H, W, C)
    im = im.transpose(1, 2, 0)
    
    # normalize range to 0 .. 1
    #im = im/2 + 0.5
    im -= im.min()
    im /= im.max()    

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    (h, w) = tuple(im.shape[:2])
    if h > maxsize:
        w = int(w * (maxsize/h)) 
        h = maxsize
        im = cv2.resize(im,(w, h))
        
    # show it in a window (this will not work on a remote session)    
    cv2.imshow(window, im)
    cv2.waitKey(100)   # display for 100 ms and wait for a keypress (which we ignore here)

name = opt.name #"out5/testcd"
steps = opt.steps
bs = 1
ifn = opt.image 

model = Unet(
    dim = 64,
    dim_mults = opt.mults # (1, 2, 4, 8)
).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size = opt.modelSize,
    timesteps = opt.sampling_steps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

diffusion.model = model

perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()
cnorm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

text = opt.text 

sk = int(1000/opt.sampling_steps)
use_timesteps = range(0, 1000, sk)
print(len(use_timesteps))
#print(list(use_timesteps))

if opt.load != "":
  data = torch.load(opt.load)
  try:
    print("loaded "+opt.load+", correct mults: "+",".join(str(x) for x in data['mults']))
  except:
    print("loaded "+opt.load+", no mults stored")

  m = "ema" if opt.ema else "model"
  dd = data[m].copy()


  # respace timesteps i.e. recalculate betas etc to match the shorter timestep schedule
  
  if opt.sampling_steps < dd['betas'].shape[0]:
      betas = dd['betas']
      alphas_cumprod = dd['alphas_cumprod']
      alphas_cumprod_prev = dd['alphas_cumprod_prev']
      sqrt_alphas_cumprod = dd['sqrt_alphas_cumprod']
      sqrt_one_minus_alphas_cumprod = dd['sqrt_one_minus_alphas_cumprod']
      
      adj_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']
      
      last_alpha_cumprod = 1.0
      new_betas = []
      for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
      betas = torch.stack(new_betas)
      
      # Use float64 for accuracy.
      #betas = np.array(betas, dtype=np.float64)
      diffusion.betas = betas
      assert len(betas.shape) == 1, "betas must be 1-D"
      assert (betas > 0).all() and (betas <= 1).all()

      #self.num_timesteps = int(betas.shape[0])

      alphas = 1.0 - betas
      diffusion.alphas = alphas
      diffusion.alphas_cumprod = torch.cumprod(alphas, axis=0)
      diffusion.alphas_cumprod_prev = F.pad(diffusion.alphas_cumprod[:-1], (1, 0), value = 1.)
      
      print(betas.shape, alphas.shape)
      
      print(diffusion.alphas_cumprod_prev.shape, diffusion.num_timesteps)
      #diffusion.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
      assert diffusion.alphas_cumprod_prev.shape == (diffusion.num_timesteps,)

      # calculations for diffusion q(x_t | x_{t-1}) and others
      diffusion.sqrt_alphas_cumprod = torch.sqrt(diffusion.alphas_cumprod)
      diffusion.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - diffusion.alphas_cumprod)
      diffusion.log_one_minus_alphas_cumprod = torch.log(1. - diffusion.alphas_cumprod)
      diffusion.sqrt_recip_alphas_cumprod = torch.sqrt(1. / diffusion.alphas_cumprod)
      diffusion.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / diffusion.alphas_cumprod - 1)

      # calculations for posterior q(x_{t-1} | x_t, x_0)
      diffusion.posterior_variance = (
          betas * (1.0 - diffusion.alphas_cumprod_prev) /
          (1.0 - diffusion.alphas_cumprod)
      )
      # log calculation clipped because the posterior variance is 0 at the
      # beginning of the diffusion chain.
      diffusion.posterior_log_variance_clipped = torch.log(diffusion.posterior_variance.clamp(min =1e-20))
      
      diffusion.posterior_mean_coef1 = betas * torch.sqrt(diffusion.alphas_cumprod_prev) / (1. - diffusion.alphas_cumprod)
      
      diffusion.posterior_mean_coef2 =(1. - diffusion.alphas_cumprod_prev) * torch.sqrt(diffusion.alphas) / (1. - diffusion.alphas_cumprod)
      
      # remove the above keys from state_dict
      
      for k in adj_keys:
         del dd[k]


  diffusion.load_state_dict(dd, strict=False)

diffusion.eval()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()])

init_noise = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1).cuda()

if ifn != "":   
  imT_ = transform(Image.open(ifn).convert('RGB')).float().cuda().unsqueeze(0)
  imT_ = (imT_ * 2) - 1
  imT = imT_ * opt.weak
  if opt.mid != 0:
      imT += opt.mid
      imT = imT.clamp(-1,1)
  mul = opt.mul
else:
   imT = init_noise
   mul = 1

if opt.tgt_image != "":   
  if opt.tgt_image == "init":
    imS = imT_.clone()
  else:
    imS = transform(Image.open(opt.tgt_image).convert('RGB')).float().cuda().unsqueeze(0)
    imS = (imS * 2) - 1

if opt.img_prompt != "":   
  imP = transform(Image.open(opt.img_prompt).convert('RGB')).float().cuda().unsqueeze(0)
  nimg = imP.clip(0,1)
  nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
  imgp_enc = perceptor.encode_image(nimg.detach()).detach()


tx = clip.tokenize(text)                        # convert text to a list of tokens 
txt_enc = perceptor.encode_text(tx.cuda()).detach()   # get sentence embedding for the tokens
del tx

@torch.enable_grad()
def cond_fn(x, t, x_s):
    global opt    
    x_is_NaN = False
    x.grad = None
    #x = x.detach().requires_grad_()
    x.requires_grad_()
    n = x.shape[0]         
    
    x_s.requires_grad_()
    x_grad = torch.zeros_like(x_s)
    
    #print(x.shape, x_s.shape, (x - x_s).max())
                
    loss = 0
    losses = []

    nimg = None

    if opt.text != "" and opt.textw > 0:
        nimg = (x_s.clip(-1, 1) + 1) / 2     
        nimg = cut(nimg, cutn=opt.cutn, low=opt.low, high=opt.high, norm = cnorm)
        
        #print(nimg.shape)
 
        # get image encoding from CLIP
 
        img_enc = perceptor.encode_image(nimg) 
  
        # we already have text embedding for the promt in txt_enc
        # so we can evaluate similarity
     
        if opt.spher:
            loss = opt.textw * spherical_dist_loss(txt_enc.detach(), img_enc).mean()
        loss = opt.textw*10*(1-torch.cosine_similarity(txt_enc.detach(), img_enc)).view(-1, bs).T.mean(1)
        losses.append(("Text loss",loss.item())) 
        if opt.tdecay < 1.:
            opt.textw = opt.tdecay * opt.textw
        #print(opt.text, loss.item())
        
        x_grad += torch.autograd.grad(loss.sum(), x_s, retain_graph = True)[0]
        #print(x_grad.shape, x_grad.std())

    if opt.img_prompt != "":
        if nimg == None:
            nimg = (x_s.clip(-1, 1) + 1) / 2     
            nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
            img_enc = perceptor.encode_image(nimg)
        loss1 = opt.imgpw*10*(1-torch.cosine_similarity(imgp_enc, img_enc)).view(-1, bs).T.mean(1)  
        losses.append(("Img prompt loss",loss1.item())) 
        loss = loss + loss1     
        
        x_grad += torch.autograd.grad(loss1.sum(), x_s, retain_graph = True)[0]
        
    if opt.tgt_image != "":
          loss_ = opt.ssimw * (1 - ssim((x_s+1)/2, (imS+1)/2)).mean() 
          losses.append(("Ssim loss",loss_.item())) 
          loss = loss + loss_    
          
          x_grad += torch.autograd.grad(loss_.sum(), x_s, retain_graph = True)[0]
  
    #loss.backward()
    #x_grad = x_grad.detach()  
    
    if torch.isnan(x_grad).any()==False:
        grad = -torch.autograd.grad(x_s, x, x_grad)[0]
    else:
      x_is_NaN = True
      grad = torch.zeros_like(x)             
          
    return grad.detach()

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predict_noise_from_start(d, x_t, t, x0):
        return (
            (extract(d.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(d.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
        
def predict_start_from_noise(self, x_t, t, noise):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )        
    
def predict_eps_from_xstart(d, x_t, t, pred_xstart):
    return (
        extract(d.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - pred_xstart
    ) / extract(d.sqrt_recipm1_alphas_cumprod, t, x_t.shape)    
    
def predict_xstart_from_eps(d, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(d.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(d.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )    
    
def scale_timesteps(t):
        return t.float() * (1000.0 / opt.sampling_steps)
        
def q_posterior_mean_variance(d, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(d.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(d.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(d.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            d.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        

@torch.enable_grad()     
def condition_mean(d, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
            
        gradient = cond_fn(x, t, p_mean_var["pred_xstart"])
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float() * opt.lr
        )
        return new_mean

def condition_score_with_grad(d, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(d.alphas_cumprod, t, x.shape)

        eps = predict_eps_from_xstart(d, x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var['pred_xstart'] #, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = predict_xstart_from_eps(d, x, t, eps)
        out["mean"], _, _ = q_posterior_mean_variance(
            d, x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

def p_mean_variance(d, x, t, clip_denoised: bool, denoise_fn=None, retdict=True):
    with torch.enable_grad():
         model_output = d.denoise_fn(x, scale_timesteps(t)) # d.denoise_fn(x, t)

         x.requires_grad_()
         x_start = d.predict_start_from_noise(x, t = t, noise = model_output)
         x_start.requires_grad_()
        
         if clip_denoised:
             x_start.clamp_(-1., 1.)

         model_mean, posterior_variance, posterior_log_variance = d.q_posterior(x_start = x_start, x_t = x, t = t)
         
         if retdict:
             out = {}
             out['pred_xstart'] = x_start
             out['mean'] = model_mean
             out["variance"] = posterior_variance
             out["pred_noise"] = model_output
             
             return out
         else:     
             return model_mean, posterior_variance, posterior_log_variance, x_start
                    
def ddim_sample(
    d,
    x,
    t,
    clip_denoised=True,
    denoise_fn=None,
    cond_fn=None,
    eta=0.0,
):
    global opt
    """
    Sample x_{t-1} from the model using DDIM.
    Same usage as p_sample().
    """

    with torch.enable_grad():        
        x = x.detach().requires_grad_()
        out_orig = p_mean_variance(
            d,
            x,
            t,
            clip_denoised=clip_denoised,
            denoise_fn=denoise_fn,
            retdict = True
            #model_kwargs=model_kwargs,
        )

        if cond_fn is not None:
            out = condition_score_with_grad(d, cond_fn, out_orig, x, t)
        else:
            out = out_orig

    out["pred_xstart"] = out["pred_xstart"].detach()
    eps = predict_eps_from_xstart(d, x, t, out["pred_xstart"])


    alpha = d.alphas_cumprod[t]
    alpha_next = d.alphas_cumprod_prev[t]

    sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha)) * torch.sqrt(1 - alpha/alpha_next)
    c = (1 - alpha_next - sigma ** 2).sqrt()
    
    # Equation 12.
    noise = torch.randn_like(x)
    mean_pred = (
        out["pred_xstart"] * torch.sqrt(alpha_next)
        + c * eps #out['pred_noise']
    )

    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )  # no noise when t == 0
    
    sample = mean_pred + nonzero_mask * sigma * noise
    
    #return {"sample": sample, "pred_xstart": out["pred_xstart"]}        
    return sample
    
j = 0


indices = list(range(opt.sampling_steps - opt.skip))[::-1] 

if opt.image != "":
    my_t = torch.ones([bs], device='cuda', dtype=torch.long).cuda() * indices[0]
    imT = diffusion.q_sample(imT, my_t, init_noise)

for i in tqdm(indices):
    t = torch.tensor([i] * bs, device='cuda').cuda()
    with torch.no_grad():     
        imT = ddim_sample(diffusion, imT, t, cond_fn=cond_fn, eta=opt.eta).detach() #cond_fn) # ['sample']    

    im = None
    if opt.saveiters or (opt.saveEvery > 0 and  j % opt.saveEvery == 0):
        im = pprocess(imT.clone().detach(), opt)
        if j > opt.saveAfter:
            save_image((im+1)/2, opt.dir+"/"+name + "-" + str(j)+".png")
   
        if opt.show:
          show_on_screen(im[0].cpu())
        
    j += 1
    
save_image((imT.clone()+1)/2, opt.dir+"/"+name+"-final.png")
im = pprocess(imT.clone().detach(), opt)
save_image((im+1)/2, opt.dir+"/"+name+"-finalp.png")
   

    









