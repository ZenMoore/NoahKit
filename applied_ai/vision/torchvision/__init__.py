import torchvision
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision import utils
import PIL.Image as Image

'''
learning material : https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision/
api : https://pytorch.org/vision/stable/index.html
'''


datapath = 'training_data.pt'

'datasets'
# same as torch.utils.data.Dataset
# different datasets may have different construction methods
# but they share :
# - keyword: variable
# - transform: function, see torchvision.transforms
# - target_transform: function

# MNIST
datasets.MNIST(datapath, train=True, transform=None, target_transform=None, download=False)

# COCO Captions
# download https://cocodataset.org/ first
# c.f. https://github.com/cocodataset/cocoapi
# todo : download COCO API
cap = datasets.CocoCaptions(root= datapath, annFile= 'json annotation file', transform=transforms.ToTensor())
print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample
print("Image Size: ", img.size())
print(target)

# COCO Detection
# download https://cocodataset.org/ first
# c.f. https://github.com/cocodataset/cocoapi
datasets.CocoDetection(root="dir where images are", annFile="json annotation file", transform=transforms.ToTensor(), target_transform=lambda x: x.split())

# LSUN
datasets.LSUN(datapath, classes='train', transform=transforms.ToTensor(), target_transform=lambda x: x.split())

# ImageFolder : self-defined
ifd = datasets.ImageFolder(root=datapath, transform=transforms.ToTensor(), target_transform=lambda x: x.split())
# ifd.classes, ifd.class_to_idx, ifd.imgs=[(img-path, class)...]
# data must be arranged by this format:
# root/dog/xxx.png
# root/dog/xxy.png
# root/dog/xxz.png
#
# root/cat/123.png
# root/cat/nsdf3.png
# root/cat/asd932_.png

# ImageNet-12
# implemented by ImageFolder
datasets.ImageNet(datapath)

# CIFAR
datasets.CIFAR10(datapath, train=True, transform=None, target_transform=None, download=False)
datasets.CIFAR100(datapath, train=True, transform=None, target_transform=None, download=False)

# STL10
datasets.STL10(datapath, split='train', transform=None, target_transform=None, download=False)

'models'
squeezenet = models.squeezenet1_0()
densenet = models.densenet_161()
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)

'transforms'
img = Image.open('img.jpg')


def __crop(img, pos, size):
    """
    :param img: input image
    :param pos: crop position of image, tuple(x, y)
    :param size: crop size of image
    :return: cropped image
    """
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size

    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1+tw, y1+th))
    return img

crop = transforms.Scale(12)
transform = transforms.Compose([transforms.toTensor(), transforms.CenterCrop(10)])
transform_lambda = transforms.Lambda(lambda img: __crop(img, (5,5), 224)),
result_1 = transform(img)
result_2 = crop(img)
result_3 = transform_lambda(img)

'utils'
utils.make_grid(result_1, nrow=8, padding=2, normalize=False, range=None, scale_each=False)
utils.save_image(result_1, datapath, nrow=8, padding=2, normalize=False, range=None, scale_each=False)










