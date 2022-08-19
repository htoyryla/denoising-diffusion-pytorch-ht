from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import noise_like

import torch
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2
from pytorch_msssim import ssim
from postproc import pprocess

'''
pip install denoising_diffusion_pytorch
'''

from cutouts import cut

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--text', type=str, default="", help='text prompt')
parser.add_argument('--image', type=str, default="", help='path to init image')
parser.add_argument('--style', type=str, default="", help='path to style image')
parser.add_argument('--img_prompt', type=str, default="", help='path to image prompt')
parser.add_argument('--tgt_image', type=str, default="", help='path to target image')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--ssimw', type=float, default=1., help='target image weight')
parser.add_argument('--sw', type=float, default=10, help='style weight')
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

parser.add_argument('--style_layers', type=int, nargs='*', default=[1, 3, 5, 9, 13, 15], help='style layers indices')
parser.add_argument('--style_scale', type=float, default=1, help='')

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
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

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
    timesteps = steps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

perceptor, clip_preprocess = clip.load('ViT-B/32', jit=False)
perceptor = perceptor.eval()
cnorm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

text = opt.text 

def shortenL(inp, fi):
    newL = torch.zeros_like(inp)[0:fi]
    i = 0
    l = len(inp)
    s = int(l/fi)
    for n in range(0, fi):
        newL[n] = inp[i]
        i += s
    print(len(newL))
    return newL

if opt.load != "":
  data = torch.load(opt.load)
  try:
    print("loaded "+opt.load+", correct mults: "+",".join(str(x) for x in data['mults']))
  except:
    print("loaded "+opt.load+", no mults stored")

  m = "ema" if opt.ema else "model"
  dd = data[m].copy()
  if opt.ignore:
    for k in data[m].keys():
      if "alphas" in k:
          #dd.pop(k)
          dd[k] = shortenL(dd[k], opt.steps)
      elif "betas" in k:
          #dd.pop(k)
          dd[k] = shortenL(dd[k], opt.steps)
      elif "posterior" in k:
          #dd.pop(k)
          dd[k] = shortenL(dd[k], opt.steps)
                       
  #print(dd.keys())
  diffusion.load_state_dict(dd, strict=False)

diffusion.eval()

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()])

if ifn != "":   
  imT_ = transform(Image.open(ifn).convert('RGB')).float().cuda().unsqueeze(0)
  imT_ = (imT_ * 2) - 1
  imT = imT_ * opt.weak
  if opt.mid != 0:
      imT += opt.mid
      imT = imT.clamp(-1,1)
  mul = opt.mul
else:
   imT = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1).cuda()
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


#---------------

xsforms = []
stylesize = (opt.imageSize, opt.imageSize)
xsforms.append(Resize(stylesize))    # resize image
xsforms.append(ToTensor())            # convert to pytorch tensor
xsforms.append(Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))            # normalize to range -1...1

# compose into a transform pipeline

style_preprocess = Compose(xsforms)

if opt.style != "":

  imgS = style_preprocess(Image.open(opt.style).convert("RGB")).to(device) #.clamp_(-2,2)

  #lr = opt.lr 

  # let us first create a VGG19 network

  vgg = models.vgg19(pretrained=True).features.to(device).eval()

  # list of suitable layers for content/style evaluation

  players = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]

  #content_idx = opt.content_layer
  style_idxs = opt.style_layers #[1, 3, 5, 9, 13, 15] 

  # add a hook to read style evaluation from selected layers

  # use a gram matrix to evaluate style (texture) instead of content

  def gram(input):
    a, b, c, d = input.size() 
    f = input.clone().reshape(a * b, c * d)  # resise F_XL into \hat F_XL
    gr = torch.mm(f, f.t())  # compute the gram product
    return gr.div(a * b * c * d)
    

  # now add the actual style hooks    

  style_grams = [None]*len(style_idxs)
  def style_hook(i):
    def hook(model, input, output):
        style_grams[i] = gram(output)
    return hook

  n = 0
  for s in style_idxs:
    style_layer = vgg[players[s]]    
    style_layer.register_forward_hook(style_hook(n))
    n += 1
     
  # now we need targets for style     
      
  # feed style image to VGG and store outputs from style hooks

  style_targets = [None]*len(style_idxs)
  o = vgg(imgS.unsqueeze(0))
  for n in range(len(style_idxs)):      
    style_targets[n] = style_grams[n].detach() #.shape        

#---------------


def cond_fn(x, t, x_s=None):
  global opt    
  with torch.enable_grad():
    x_is_NaN = False
    x.grad = None
    x = x.detach().requires_grad_()
    n = x.shape[0]         
    #x_s = x_s.requires_grad_()
                
    loss = 0
    losses = []

    nimg = None

    if opt.text != "":
        nimg = (x.clip(-1, 1) + 1) / 2     
        nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
 
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

    if opt.img_prompt != "":
        if nimg == None:
            nimg = (x.clip(-1, 1) + 1) / 2     
            nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
            img_enc = perceptor.encode_image(nimg)
        loss1 = opt.imgpw*10*(1-torch.cosine_similarity(imgp_enc, img_enc)).view(-1, bs).T.mean(1)  
        losses.append(("Img prompt loss",loss1.item())) 
        loss = loss + loss1     

    if opt.tgt_image != "":
          loss_ = opt.ssimw * (1 - ssim((x+1)/2, (imS+1)/2)).mean() 
          losses.append(("Ssim loss",loss_.item())) 
          loss = loss + loss_    
          
    if opt.style != "":
        o = vgg(x)

        # store style actuals

        style_actuals = [None]*len(style_idxs)

        # evaluate total style loss

        losss = torch.tensor([0.]).cuda()
        if opt.sw > 0:
          for n in range(len(style_idxs)):      
            style_actuals[n] = style_grams[n]
            sl = opt.sw * F.mse_loss(style_targets[n], style_actuals[n])
            losss += sl
        loss_ = losss.mean()           
        losses.append(("Style loss", loss_.item()))
        loss = loss + loss_
          
    loss.backward()
    x_grad = x.grad.detach()  
    #x_grad = torch.autograd.grad(loss, x)[0]
    #print(x_grad.min(), x_grad.max())
    #print(x.grad)      
    
    #print(x_grad.shape, x.shape, x_s.shape)
    #if torch.isnan(x_grad).any() == True:
    #    x_grad = torch.zeros_like(x)                   
          
  return x_grad, losses

  
def p_mean_variance(d, x, t, clip_denoised: bool, denoise_fn=None):
    with torch.enable_grad():
         model_output = denoise_fn(x, t) # d.denoise_fn(x, t)

         #if self.objective == 'pred_noise':
         #x_start = model_output 
         x_start = d.predict_start_from_noise(x, t = t, noise = model_output)
         #print(x_start.shape, x.shape)
         #elif self.objective == 'pred_x0':
         #x_start = model_output
         #else:
         #    raise ValueError(f'unknown objective {self.objective}')

         if clip_denoised:
             x_start.clamp_(-1., 1.)

         model_mean, posterior_variance, posterior_log_variance = d.q_posterior(x_start = x_start, x_t = x, t = t)
         return model_mean, posterior_variance, posterior_log_variance, x_start
   

def p_sample_with_cond(d, x, t, cond_fn=None, clip_denoised=True, repeat_noise=False):
        global diffusion
        b, *_, device = *x.shape, x.device
        #model_mean, var, model_log_variance = diffusion.p_mean_variance(x, t, clip_denoised=clip_denoised)
        model_mean, var, model_log_variance, x_s = p_mean_variance(d, x, t, clip_denoised=clip_denoised,  denoise_fn=d.denoise_fn)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        if cond_fn:
          grad, losses = cond_fn(x, t) #, x_start)
          #print(grad.shape, grad.min(), grad.max())
          new_mean = model_mean - var * grad * opt.lr       
          out = new_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        else:
          out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
          losses = []
         
        
        return out.detach(), losses 

j = 0
for i in tqdm(reversed(range(opt.skip, steps)), desc='sampling loop time step', total=steps): 
    t = torch.full((bs,), i // mul, device='cuda', dtype=torch.long).cuda()
    
    imT, losses = p_sample_with_cond(diffusion, imT.detach(), t, cond_fn=cond_fn)
    
    im = None
    if opt.saveiters or (opt.saveEvery > 0 and  j % opt.saveEvery == 0):
        im = pprocess(imT.clone().detach(), opt)
        if j > opt.saveAfter:
            save_image((im+1)/2, opt.dir+"/"+name + "-" + str(j)+".png")
   
        if opt.show:
          show_on_screen(im[0].cpu())
        
        if opt.showLosses:
              out = ""
              for item in losses:
                out += item[0] + ":" + str(item[1]) + " "
              print(out)
        
    j += 1
    
save_image((imT.clone()+1)/2, opt.dir+"/"+name+"-final.png")
im = pprocess(imT.clone().detach(), opt)
save_image((im+1)/2, opt.dir+"/"+name+"-finalp.png")
   

    









