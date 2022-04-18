# adapted from
# https://github.com/lucidrains/deep-daze/blob/9435bcb903044a90320d9676b70e4909d82c827d/deep_daze/deep_daze.py

# prepares an image for CLIP encoder by making cutouts in various scales
# and resizing them all into 224x224

import torch
import torch.nn.functional as F

# use torch interpolation for tensor resize

def interpolate(image, size):
    return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)

# make one random cutout of given size

def rand_cutout(image, size):
    width = image.shape[-1]
    offsetx = torch.randint(0, width - size, ())
    offsety = torch.randint(0, width - size, ())
    cutout = image[:, :, offsetx:offsetx + size, offsety:offsety + size]
    return cutout

def cut(out, cutn=8, low=0.4, high=1.0, norm=None):
    
        # sample cutout sizes between lower and upper bound
        width = out.shape[-1]
        lower = low * width
        upper = high * width
        sizes = torch.randint(int(lower), int(upper), (cutn,))
        #print(sizes)

        # create random cutouts according to the list of sizes created above and resize them to 224px
        if norm is not None:
            pieces = torch.cat([norm(interpolate(rand_cutout(out, size), 224)) for size in sizes])
        else:
            pieces = torch.cat([interpolate(rand_cutout(out, size), 224) for size in sizes])            
            
        return pieces

