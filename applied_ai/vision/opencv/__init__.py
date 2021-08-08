import cv2
import numpy as np
import time
import argparse

'''
learning material 1 introduction : https://blog.csdn.net/abc123mma/article/details/111309122?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1.pc_relevant_baidujshouduan&spm=1001.2101.3001.4242
learning material 2 operations & application : http://www.woshicver.com/ 
api : http://www.woshicver.com/
'''

'''
github : https://github.com/opencv/opencv
there are many models and tools.
compared to pillow, opencv is much more powerful.
'''

img_path = '../../../asset/img.jpg'

'image reading'
img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)  # UNCHANGED: +alpha channel

'image show and storage'
cv2.imshow('this is an image window', img)  # we can also use matplotlib plt.imshow
# >0: wait >0 seconds, if press a key, return the key, if not return -1; =0: endless waiting; none: flash
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('new_img.jpg', img)
cv2.destroyAllWindows()  # don't forget

'plotting'
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)  # rectangle, triangle, etc.
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'image of a beauty', (100, 100), font, 2.5, (0, 0, 225), 2)

'interaction'
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

cv2.namedWindow('image')  # a named window, attention, name is not the window's title, but the variable ID
cv2.setMouseCallback('image', draw_circle)
while (True):
    cv2.imshow('image window', img)
    # & is bit-and, 0xFF is 11111111
    # by this bit operation, we leave 8-bit of waitkey, we need this because waitkey is ASCII
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()

'image operation'
# flip
cv2.flip(img, flipCode=0)  # =0: along x; >0: along y; <0 along x and y
# color conversion
cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# pixel editing
pixel = img[100, 100]  # if one pixel, better use : img.item(100, 100)
blue = img[100, 100, 0]
img.itemset((100, 100, 0), 0)
# RGB split and merge
b, g, r = cv2.split(img)
img = cv2.merge((b, g, 4))
# add border
replicate = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
# binarization
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# background and foreground (using convolution)
threshold = 0.2
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)  # morphological operation
bg = cv2.dilate(opening, kernel, iterations=10)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # ret=True is the pic is available
fg = np.uint8(fg)
unknown = cv2.subtract(bg, fg)

'with numpy'
img = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
], dtype=np.uint8)
cv2.imwrite('img_cv2', img)  # plt.save('img_plt', img)

'video'
# for video : VideoCapture and VideoWrite

interval = 60  # video capture time interval between two frames
num_frames = 500
out_fps = 24

cap = cv2.VideoCapture(0)  # open default camera
# if we want to capture video from existing video file
# cap = cv2.VideoCapture()
# cap.open(filepath)
# if we want to set capture configuration
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video = cv2.VideoWriter(
    'time.avi',
    cv2.VideoWriter_fourcc('M', 'P', '4', '2'),
    out_fps,
    size
)

# some low cameras may not be stable, so skip these first frames
for i in range(42):
    cap.read()

try:
    for i in range(num_frames):
        _, frame = cap.read()
        video.write(frame)
        print('Frame {} is captured.'.format(i))
        time.sleep(interval)
except KeyboardInterrupt:
    print('Stopped! {}/{} frames captured!'.format(i, num_frames))

video.release()
cap.release()

# get video num_frames
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

'ml model'
# there are many machine learning models embeded in cv2
# but we think that sklearn is a better choice
# we use cv2 to do something that is more about image processing
# the models of cv2 that we use frequently are more about image processing, too
# for example, SIFT, Hough Transform algorithm, GrabCut algorithm, etc.
knn = cv2.ml.KNearest_create()
# knn.train(...), knn.findNearest(...)

'''
CascadeClassifier
application : face detection
'''

# we use cascaded classifier in cv2 to detect face and eyes
# for more models like this, please view api : http://www.woshicver.com/
# we should download the .xml first : https://github.com/opencv/opencv/tree/master/data/haarcascades
# and more tools are in https://github.com/opencv/opencv
def detectAndDisplay(frame):
    frame_gray = cv2.cv2tColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]
        # -- detect eyes on each face
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv2.imshow('Capture - Face detection', frame)


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                    default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()
# -- 1. load cascade
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
# -- 2. read video flow
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()  #
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv2.waitKey(10) == 27:
        break
