# Where's Waldo?

### VGG template matching

A Pytorch implementing of [A Deep Learning approach to Template Matching](http://cs231n.stanford.edu/reports/2017/pdfs/817.pdf)

Require one template image and one source image.

Template image needs to be **padded** to the same size as source image.

And then resize to (512, 512) ...

```python
from model import TemplateMatching

t = torch.randn(4, 3, 512, 512).cuda()
x = torch.randn(4, 3, 512, 512).cuda()
net = TemplateMatching(
    z_dim=64, output_channel=512, pretrain=False).cuda()
    
# checkpoint
checkpoint = torch.load('path/to/checkpoint/model_best.pth) # output/checkpoint/
net.load_state_dict(checkpoint['TemplateMatching'])
res = net(x, t) # binary map 
```

### CCOEFF template matching (seudo label)

Require one source image and a folder containing all of the probable apperences of target.

Return the location of target (only the most likely one).

```python
from ccoeff import template_matching

template_dir = 'data/templates/pacman/'
img = 'source.png'
res = template_matching(img, template_dir,
                            vis=False, return_ori=False)
```
