# Clog Loss Video Classification

Codes to Driven Data's [Clog Loss: Advance Alzheimer’s Research with Stall Catchers](https://www.drivendata.org/competitions/65/clog-loss-alzheimers-research/page/207/) challenge. The goal is to predict whether the outlined vessel segment in each video is stalled or not.

## Data Description

#### Dataset given:

| Training Dataset Version  | Size | Class ratio (flowing / stalled) |
| ------------- | ------------- | ------------- |
| Nano  | 	3.8 GB  | 50 / 50 |
| Micro  | 6.4 GB  | 70 / 30 |
| Full | 1.4 TB  | 99.7 / 0.3 |

- Videos are in mp4 file format.
- Test set size is 35.4 GB with 14,160 videos.

#### Performance metric

The test set result is evaluated against Matthew's correlation coefficient (MCC) which takes into account true positives, true negatives, false positives, and false negatives. 
It ranges between -1 and 1, with 1 representing perfect prediction, 0 as no better than random, and -1 a perfect disagreement between predicted and observed values.

I scored 0.1425, with a rank of 46 / 922 participants.

## Approach

I used the Micro training dataset to train the model.

1. Get frames of videos using OpenCV's `VideoCapture` - [prepare_data.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/prepare_data.ipynb)
2. Draw a bounding box over the outlined vessel and crop image using OpenCV's `boundingRect` - [crop_data.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/crop_data.ipynb)
3. Use PyTorch's pre-trained ResNet 3D model to train a model on our dataset - [3DCNN.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/3DCNN.ipynb)
4. Test model on test dataset - [test_model.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/test_model.ipynb)

### 0. Install required libraries

- OpenCV - I used OpenCV for Windows.
- PyTorch - refer to the [documentation guide](https://pytorch.org/). I used CUDA 10.2 on Windows. 

```
pip install opencv-python==4.3.0.36
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### 1. Capture video frames

Notebook: [prepare_data.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/prepare_data.ipynb)

For each video, 16 frames are captured and saved. These frames are equally spaced over time in the video.

```
def get_frames(filename, n_frames= 1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, v_len-1, n_frames+1, dtype=np.int16)
    
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame)
    v_cap.release()
    return frames, v_len
 ```
 
 ### 2. Crop image to outlined vessel
 
 Notebook: [crop_data.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/crop_data.ipynb)
 
 Since the area of interest is the outlined vessel (circled in red), the other areas in the picture may pose as a distraction, especially when blood moves over time. 
 Therefore, I cropped out the outlined vessel and save this image.
 
![alt text](https://github.com/agrilive/clog-loss-video-classification/blob/master/sample_image/micro_jpg_stalled_frame0.jpg?raw=true) _Example frame of a stalled vessel_

**Identify the red colours in the image:**

```
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

mask = cv2.bitwise_or(mask1, mask2)
```

![alt text](https://github.com/agrilive/clog-loss-video-classification/blob/master/sample_image/micro_jpg_stalled_mask_frame0.jpg?raw=true)

**Draw a bounding box:**

```
x,y,w,h = cv2.boundingRect(mask)
mask_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
```

![alt text](https://github.com/agrilive/clog-loss-video-classification/blob/master/sample_image/micro_jpg_stalled_bbox_frame0.jpg?raw=true)

**Crop image:**

```
roi = img[y:y+h, x:x+w]
```

![alt text](https://github.com/agrilive/clog-loss-video-classification/blob/master/sample_image/micro_jpg_crop_stalled_frame0.jpg?raw=true)

As OpenCV identifies the region of interest differently for each frame, I standardised the ROI of all the frames in a video by using the ROI of the first frame (frame 0).

### 3. Train model

Notebook: [3DCNN.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/3DCNN.ipynb)

Since we are identifying whether the vessel is stalled or not, it is important to know whether the blood flows over time. Therefore, we use a 3D CNN instead of a 2D CNN. The 3D CNN has an additional input dimension - time - compared to the 2D CNN. In other words, the frames of a video are stacked and loaded into the model.

![alt text](https://github.com/agrilive/clog-loss-video-classification/blob/master/sample_image/micro_jpg_crop_unstalled_frames.JPG?raw=true) _Example frames of an unstalled vessel_

#### Image augmentation

Image augmentation is used before loading the frames into the model. 

```
train_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
```

For a 3D CNN, it is important for the same augmentation to applied to all frames of a video. Therefore, a random seed is generated before each transformation.

```
seed = np.random.randint(1e9)        
frames_tr = []
for frame in frames:
    random.seed(seed)
    np.random.seed(seed)
    frame = self.transform(frame)
    frames_tr.append(frame)
```

#### ResNet 3D

The model used here is a pre-trained ResNet 3D model. 

```
from torchvision import models
from torch import nn

model = models.video.r3d_18(pretrained=True, progress=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
```

I trained for 10000 epochs, which took about 8 hours on a GPU (RTX 2070 SUPER).

### 4. Test model

Notebook: [test_model.ipynb](https://github.com/agrilive/clog-loss-video-classification/blob/master/test_model.ipynb)

The same pre-processing steps were applied to the test dataset - capturing the frames of the video, cropping out the region of interest. The pre-processed frames were then used for prediction.

## Reflections

This was my first time coming across a video dataset. I spent quite a long time figuring out how to approach the problem. In fact, I started out using a pre-trained 2D CNN and scored a 0. Although my results did not meet my target, I felt that the learnings from this process were huge.

Firstly, I tried dabbling with the Tensorflow tf.data pipeline. It was tough trying to stack the frames by each video and loading them into memory. This was salvaged by using PyTorch DataLoader instead.

I tried training the model on the uncropped images and the results were not satisfactory. That was when I had the idea of cropping out the ROI. It may not work on another dataset if the colors are not distinct (the images here were black and white, with red markings of the ROI). 

Given the limited time, I only managed to train a Conv3d model on the Micro training dataset. Would love to have tried other techniques such as Conv-LSTM, using other pre-trained 3D CNN models and adding more layers to the pre-trained ResNet 3D model. Not to mention, I would love to try out the 1.4 terabytes dataset, which is an imbalanaced dataset and hosted on a public s3 bucket. 

