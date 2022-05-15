from kornia.enhance import adjust_contrast, adjust_brightness, adjust_saturation, adjust_gamma
from kornia.filters import unsharp_mask, median_blur
from kornia.enhance.equalization import equalize_clahe
import cv2
import numpy as np
import torch

def pprocess(imT, opt):

        # normalize to 0..1 (needed for contrast etc adjustments)

        imT -= imT.min()
        imT /= imT.max()

        if opt.contrast != 1:
            imT = adjust_contrast(imT, opt.contrast)
        if opt.gamma != 1:
            imT = adjust_gamma(imT, opt.gamma)
        if opt.brightness != 0:
            imT = adjust_brightness(imT, opt.brightness)
        if opt.saturation != 1:
            imT = adjust_saturation(imT, opt.saturation)
 
       # usually the histogram is narrow, optionally equalize to get a wider histogram
            
        imT_ = imT.clone()            

        if opt.median > 0:
            #imT = median_blur(imT, (opt.median, opt.median)) 
            # alternative implementation of median blur using opencv
            # to get rid of out of memory crashes using kornia
            # this is a hack: converts from tensor to opencv and back 
           
            img = imT.clone()[0].cpu().permute(1,2,0).numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.medianBlur(img, opt.median)/255.
            imT = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
        elif opt.bil > 0:
            # alternative implementation of median blur using opencv
            # to get rid of out of memory crashes using kornia
            # this is a hack: converts from tensor to opencv and back 
            
            img = imT.clone()[0].cpu().permute(1,2,0).numpy()
            img = (img*255).astype(np.uint8)
            img = cv2.bilateralFilter(img, opt.bil, opt.bils1, opt.bils2)/255.
            imT = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()    
            
            
        if opt.unsharp > 0:
            imT = unsharp_mask(imT, (opt.sharpkernel, opt.sharpkernel), (opt.unsharp, opt.unsharp))    

        if opt.ovl0 != 0:
            imT = opt.ovl0 * imT_ + (1 - opt.ovl0) * imT


        imT -= imT.min()
        imT /= imT.max()   

        if opt.eqhist > 0:
            imT = equalize_clahe(imT, clip_limit = opt.eqhist, grid_size = (8,8))
            
        if opt.sharpenlast and opt.unsharp > 0:
            imT = unsharp_mask(imT, (opt.sharpkernel, opt.sharpkernel), (opt.unsharp, opt.unsharp))

        imT = 2 * imT
        imT = imT - 1

        return imT   

