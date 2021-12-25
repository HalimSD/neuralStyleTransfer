import torch
from torch._C import device
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ToTensor()
])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_image = image_loader('./images/picasso.jpg')
content_image = image_loader('./images/dancing.jpg')

assert style_image.size() == content_image.size(), "import style and content images of the same size"

unloader = transforms.ToPILImage() # reconvert the tensor image to pil to display it
plt.ion() # turn the interactive mode on

def im_show(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title = title
    plt.pause(0.001)

plt.figure()
im_show(content_image, 'Content Image')

plt.figure()
im_show(style_image, 'Style Image')

