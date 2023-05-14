from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import noise_like
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import clip
import argparse
import cv2
from pytorch_msssim import ssim
from postproc import pprocess
import random

'''
pip install denoising_diffusion_pytorch
'''

from cutouts import cut

parser = argparse.ArgumentParser()

# define params and their types with defaults if needed
parser.add_argument('--text', type=str, default="", help='text prompt')
parser.add_argument('--image', type=str, default="", help='path to image')
parser.add_argument('--img_prompt', type=str, default="", help='path to image')
parser.add_argument('--tgt_image', type=str, default="", help='path to image')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--ssimw', type=float, default=1., help='ssim weight')
parser.add_argument('--textw', type=float, default=1., help='text weight')
parser.add_argument('--tdecay', type=float, default=1., help='text weight decay')
parser.add_argument('--imgpw', type=float, default=1., help='image prompt weight')
parser.add_argument('--steps', type=int, default=1000, help='number of iterations')
parser.add_argument('--skip', type=int, default=0, help='number of iterations')
parser.add_argument('--skipend', type=int, default=0, help='number of iterations')
parser.add_argument('--dir', type=str, default="out", help='basename for storing images')
parser.add_argument('--name', type=str, default="test", help='basename for storing images')
parser.add_argument('--mul', type=float, default=1., help='')
parser.add_argument('--show', action="store_true", help='show image in a window')
parser.add_argument('--ema', action="store_true", help='')

parser.add_argument('--canvasSize', type=int, default=2048, help='image size')
parser.add_argument('--h', type=int, default=0, help='image size')
parser.add_argument('--w', type=int, default=0, help='image size')
parser.add_argument('--modelSize', type=int, default=512, help='image size')

parser.add_argument('--tilemin', type=int, default=512, help='image size')
parser.add_argument('--tilemax', type=int, default=1024, help='image size')
parser.add_argument('--tiles', type=int, default=64, help='image size')

parser.add_argument('--saveEvery', type=int, default=0, help='image save frequency')
parser.add_argument('--saveAfter', type=int, default=0, help='image save frequency')
parser.add_argument('--low', type=float, default=0.4, help='lower limit for cut scale')
parser.add_argument('--high', type=float, default=1.0, help='higher limit for cut scale')
parser.add_argument('--cutn', type=int, default=24, help='number of cutouts for CLIP')
parser.add_argument('--load', type=str, default="", help='path to pth file')
#parser.add_argument('--saveiters', action="store_true", help='show image in a window')
#parser.add_argument('--saveLat', type=str, default="", help='path to save pth')
parser.add_argument('--mults', type=int, nargs='*', default=[1, 1, 2, 2, 4, 8], help='')
parser.add_argument('--weak', type=float, default=1., help='weaken input img')
parser.add_argument('--mid', type=float, default=0, help='weaken init image')
parser.add_argument('--model', type=str, default="", help='')
parser.add_argument('--gradv', action="store_true", help='')
parser.add_argument('--showLosses', action="store_true", help='display losses')
parser.add_argument('--spher', action="store_true", help='use spherical loss')

parser.add_argument('--contrast', type=float, default=1, help='contrast, 1 for neutral')
parser.add_argument('--brightness', type=float, default=0, help='brightness, 0 for neutral')
parser.add_argument('--saturation', type=float, default=1, help='saturation, 1 for neutral')
parser.add_argument('--gamma', type=float, default=1, help='gamma, 1 for neutral')
parser.add_argument('--unsharp', type=float, default=0, help='use unsharp mask')
parser.add_argument('--eqhist', type=float, default=0., help='histogram eq level')
parser.add_argument('--median', type=int, default=0, help='median blur, 0 for none')
parser.add_argument('--c1', type=float, default=0., help='min level adj')
parser.add_argument('--c2', type=float, default=1., help='max level adj')
parser.add_argument('--sharpenlast', action="store_true", help='save input into file')
parser.add_argument('--sharpkernel', type=int, default=3, help='median blur, 0 for none')
parser.add_argument('--ovl0', type=float, default=0, help='')
parser.add_argument('--bil', type=int, default=0, help='')
parser.add_argument('--bils1', type=int, default=75, help='')
parser.add_argument('--bils2', type=int, default=75, help='')

parser.add_argument('--savelatest', action="store_true", help='')
parser.add_argument('--updateTarget', action="store_true", help='')
parser.add_argument('--postproc', action="store_true", help='')
parser.add_argument('--grid', action="store_true", help='')

parser.add_argument('--saveIters', type=int, nargs='*', default=[], help='')




opt = parser.parse_args()

mtype = opt.model

if opt.h == 0:
    opt.h = opt.canvasSize

if opt.w == 0:
    opt.w = opt.canvasSize
    

if mtype == "unet0":
  from alt_models.Unet0 import Unet
elif mtype == "unet1":
  from alt_models.Unet1 import Unet
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
ifn = opt.image #"/work/dset/hf2019/train/DSC00131.JPG" #20150816_144314-a1.png"
#isize = opt.imageSize

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

text = opt.text #"a portrait of frightened Dostoevsky, a watercolor with charcoal"

if opt.load != "":
  data = torch.load(opt.load)

  try:
    print("loaded "+opt.load+", correct mults: "+",".join(str(x) for x in data['mults']))
  except:
    print("loaded "+opt.load+", no mults stored")

  m = "ema" if opt.ema else "model"
  diffusion.load_state_dict(data[m], strict=False)

transform = transforms.Compose([transforms.Resize((opt.h, opt.w)), transforms.ToTensor()])

if ifn != "":   
  imTf_ = transform(Image.open(ifn).convert('RGB')).float().unsqueeze(0)
  imTf_ = (imTf_ * 2) - 1
  imTf = imTf_*opt.weak
  mul = opt.mul
else:
   imTf = torch.zeros(bs,3,opt.h,opt.w).normal_(0,1) #.cuda()
   mul = 1

if opt.tgt_image != "":
  if opt.tgt_image == "init":
     imSf = imTf_.clone() 
  else:  
    imSf = transform(Image.open(opt.tgt_image).convert('RGB')).float().unsqueeze(0)
    imSf = (imSf * 2) - 1

if opt.img_prompt != "":   
  imP = transform(Image.open(opt.img_prompt).convert('RGB')).float().cuda().unsqueeze(0)
  #imP = (imP * 2) - 1
  #nimg = (imP.clip(-1, 1) + 1) / 2     
  nimg = imP.clip(0,1)
  nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
  imgp_enc = perceptor.encode_image(nimg.detach()).detach()


tx = clip.tokenize(text)                        # convert text to a list of tokens 
txt_enc = perceptor.encode_text(tx.cuda()).detach()   # get sentence embedding for the tokens
del tx

def tilexy():
    th = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    tw = (random.randint(opt.tilemin, opt.tilemax)//64)*64
    ty = random.randint(0, opt.h - th)
    tx = random.randint(0, opt.w - tw)
    #print(ty, tx, th, tw)
    return (ty, tx, th, tw)
    
def tileList():
    tlist = []
    if opt.grid:
        size = opt.tilemin
        nx = opt.w // size
        ny = opt.h // size
        tx = 0
        print(nx, ny)
        for ix in range(0, nx):
            ty = 0
            for iy in range(0, ny):
              tlist.append((ty, tx, size, size))
              ty += size
            tx += size
    else:
        for i in range(0, opt.tiles):
            tlist.append(tilexy())                          
    random.shuffle(tlist)
    return tlist

def getTile(field, pos):
    ty, tx, th, tw = pos
    tile = field[:, :, ty:ty+th, tx:tx+th]
    return tile
    
def putTile(field, pos, content):
    ty, tx, th, tw = pos
    field[:, :, ty:ty+th, tx:tx+th] = content
    return field
        
def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)             
        
def cond_fn(x, t, x_s):
  global opt    
  with torch.enable_grad():
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
        nimg = (x.clip(-1, 1) + 1) / 2     
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
        
        x_grad += torch.autograd.grad(loss.sum(), x, retain_graph = True)[0]
        #print(x_grad.shape, x_grad.std())

    if opt.img_prompt != "":
        if nimg == None:
            nimg = (x.clip(-1, 1) + 1) / 2     
            nimg = cut(nimg, cutn=12, low=0.6, high=0.97, norm = cnorm)
            img_enc = perceptor.encode_image(nimg)
        loss1 = opt.imgpw*10*(1-torch.cosine_similarity(imgp_enc, img_enc)).view(-1, bs).T.mean(1)  
        losses.append(("Img prompt loss",loss1.item())) 
        loss = loss + loss1     
        
        x_grad += torch.autograd.grad(loss1.sum(), x, retain_graph = True)[0]
        print(x_grad.shape, x_grad.std())

    if opt.tgt_image != "":
          loss_ = opt.ssimw * (1 - ssim((x+1)/2, (imS+1)/2)).mean() 
          losses.append(("Ssim loss",loss_.item())) 
          loss = loss + loss_    
          
          x_grad += torch.autograd.grad(loss_.sum(), x, retain_graph = True)[0]
  
    #loss.backward()
    #x_grad = x.grad.detach()  
    
    if torch.isnan(x_grad).any()==False:
        grad = torch.autograd.grad(x_s, x, x_grad)[0]
    else:
      x_is_NaN = True
      grad = torch.zeros_like(x)             
          
  return grad, losses
  
def p_mean_variance(d, x, t, clip_denoised: bool, denoise_fn=None):
    with torch.enable_grad():
         model_output = denoise_fn(x, t) # d.denoise_fn(x, t)

         x_start = d.predict_start_from_noise(x, t = t, noise = model_output)

         if clip_denoised:
             x_start.clamp_(-1., 1.)

         model_mean, posterior_variance, posterior_log_variance = d.q_posterior(x_start = x_start, x_t = x, t = t)
         return model_mean, posterior_variance, posterior_log_variance, x_start
   
def p_sample_with_cond(d, x, t, cond_fn=None, clip_denoised=True, repeat_noise=False):
        global diffusion
        b, *_, device = *x.shape, x.device
        #model_mean, var, model_log_variance = diffusion.p_mean_variance(x, t, clip_denoised=clip_denoised)
        with torch.enable_grad():
          x.requires_grad_()    
          model_mean, var, model_log_variance, x_s = p_mean_variance(d, x, t, clip_denoised=clip_denoised,  denoise_fn=d.denoise_fn)
          noise = noise_like(x.shape, device, repeat_noise)
          # no noise when t == 0
          nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
          x_s.requires_grad_()
          if cond_fn:
            grad, losses = cond_fn(x, t, x_s)
            #print(grad.shape, grad.min(), grad.max())
            new_mean = model_mean - var * grad * opt.lr       
            out = new_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
          else:
            out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            losses = []
         
        
        return out.detach(), losses 


        
save_image((imTf.clone()+1)/2, opt.dir+"/"+name+"-init.png")

tilelist = tileList()

print(tilelist)

imTf_ = {}
for k in opt.saveIters:
    imTf_[str(k)] = imTf.clone()

for tn in range(0, len(tilelist)):
  j = 0
  tile = tilelist[tn] #tilexy(opt.h, opt.w)
  imT = getTile(imTf, tile).cuda()

  if opt.tgt_image != "":   
     imS = getTile(imSf.clone(), tile).cuda() 
  
  textw = opt.textw

  for i in tqdm(reversed(range(opt.skip, steps)), desc='sampling loop time step', total=steps): 
    t = torch.full((bs,), i // mul, device='cuda', dtype=torch.long).cuda()
    
    imT, losses = p_sample_with_cond(diffusion, imT.clone().detach(), t, cond_fn=cond_fn)
    
    im = None
    if  j in opt.saveIters:
        print(i, j, tn, imTf_.keys())
        #im = pprocess(imT.clone().cpu().detach(), opt)
        im = imT.clone().cpu().detach()
        
        imTf_[str(j)] = putTile(imTf_[str(j)], tile, im.cpu().detach()) 
        save_image((imTf_[str(j)].clone()+1)/2, opt.dir+"/"+name+"-"+str(j)+"-"+str(tn)+"-full.png")
             
        if opt.show:
          show_on_screen(im[0].cpu().detach())
        
        if opt.showLosses:
              out = ""
              for item in losses:
                out += item[0] + ":" + str(item[1]) + " "
              print(out)
        
        
    j += 1
    
  save_image((imT.clone()+1)/2, opt.dir+"/"+name+"-tile-"+str(tn)+".png")
  
  if opt.postproc:
    imT = pprocess(imT.clone().detach(), opt)
      
  imTf = putTile(imTf, tile, imT.cpu().detach()) 
  save_image((imTf.clone()+1)/2, opt.dir+"/"+name+"-"+str(tn)+"finalp.png")

  if opt.updateTarget:
      # update target 
      imSf = putTile(imSf, tile, imT.cpu().detach())

  if opt.savelatest:
        save_image((imTf.clone()+1)/2, "/var/www/html/latest_.jpg") 
        os.rename('/var/www/html/latest_.jpg','/var/www/html/latest.jpg')   
    










