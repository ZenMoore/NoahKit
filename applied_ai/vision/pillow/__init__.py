import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance
from PIL import ImageSequence

'''
learning material 1 introduction : https://blog.csdn.net/salove_y/article/details/78824278
learning material 2 official tutorial : https://pillow.readthedocs.io/en/stable/handbook/tutorial.html
api : https://pillow.readthedocs.io/en/stable/
'''

# for installation, the package name is strange, we use :
# pip install pillow

img_path = '../../../asset/img.jpg'

'image io'
img = Image.open(img_path)
w, h = img.size
# img.save('new_img.jpg', 'jpeg')

'image operation'
# scale
img.thumbnail((w//2, h//2))
# blurring
img.filter(ImageFilter.BLUR)
# draw
image = Image.new('RGB', (240, 60), (255, 255, 255))
font = ImageFont('Arial.tff', 36)
painter = ImageDraw.Draw(image)
painter.point((10, 10), fill=(25, 35, 12))  # line, circle, ...
painter.text((20, 20), 'hello', font=font, fill=(24, 15, 245))
image.save('new_image.jpg', 'jpeg')
# crop, transpose and paste
box = (100, 100, 400, 400)
region = img.crop(box)
region = region.transpose(Image.ROTATE_180)  # rotate(),resize()...
img.paste(region, box)
# pixel operation
img.point(lambda i: i*1.2)
# contrast enhancement
enh = ImageEnhance.Contrast(img)
enh.enhance(1.3).show("30% more contrast")
# rgb split and merge
# the same as opencv

'image sequence'
# this is to say gif
with Image.open('animation.gif') as im:
    im.seek(1)  # the second frame
    for frame in ImageSequence.Iterator(im):
        pass

'video'
# we use opencv, hence, opencv is more powerful !