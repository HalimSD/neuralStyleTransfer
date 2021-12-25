import torch
from torch._C import device
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ContentLoss(nn.Module):
  def __init__(self, target,):
    super(ContentLoss, self).__init__()
    self.target = target.detch()

  def forward(self, input):
    self.loss = nn.functional.mse_loss(input, self.target)
    return input

def gram_matrix(input):
  batch_size, feature_maps_numbers, dim1, dim2 = input.size()
  features = input.view(batch_size * feature_maps_numbers, dim1 * dim2)
  G = torch.mm(features, features.t())
  return G.div(batch_size, feature_maps_numbers, dim1, dim2)

class StyleLoss(nn.Module):
  def __init__(self, target_features):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_features).detach()

  def forward(self, input):
    G = gram_matrix(input)
    self.loss = nn.functional.mse(G, self.target)
    return input

cnn = models.vgg16(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
  def __init__(self, mean, std):
    super().__init__(Normalization, self)
    self.mean = torch.tensor(mean).view(-1,1,1)
    self.std = torch.tensor(std).view(-1,1,1)

  def forward(self, img):
    return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layer_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_content_and_losses(cnn, normalization_mean, normalization_std, 
                                 style_image, content_image, 
                                 content_layers = content_layers_default, style_layers = style_layer_default):
  
  normalization = Normalization(normalization_mean, normalization_std).to(device)
  content_losses = []
  style_losses = []
  model = nn.Sequential(normalization)

  i = 0
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
      name = 'relu_{}'.format(i)
      layer = nn.ReLU(inplace=False)
    elif isinstance|(layer, nn.MaxPool2d):
      name = 'pool_{}'.formate(i)
    elif isinstance(layer, nn.BatchNorm2d):
      name = 'bn_{}'.format(i)
    else: 
      raise RuntimeError('unrecognized layer: {}'.format(layer.__class__.__name__))
    
    model.add_module(name, layer)

    if name in content_layers:
      target = model(content_image).detach()
      content_loss = ContentLoss(target=target)
      model.add_module("content_loss_{}".format(i), content_loss)
      content_losses.append(content_loss)

    if name in style_layers:
      target_features = model(style_image).detach()
      style_loss = StyleLoss(target_features=target_features)
      model.add_module("style_loss_{}".format(i), style_loss)
      style_losses.append(style_loss)
  
  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
      break
  
  model = model[:(i+1)]
  return model, style_losses, content_losses