# %% [markdown]
# # LAB5: YOLO implementation
# ## Based on the given code, write a Python program using the OpenCV, TensorFlow and Keras libraries that performs the following functions:
# 1. Implement the YOLO algorithm in the version of your choise.
# 2. On the photo indicated in the commandline, the algorithm will indicate the detected objects and mark the on the screen.
# 3. On the film indicated in the commandline it will indicate the detected objects and mark the on the screen.
# 4. The object should be marked as an appropriate envelope and class symbol throughout all the duration of the movie.

# ### Please send the program code and the results of the code (photo and video with tagged objects) via Teams at the time specified in the assignment.
# ### Good luck
# #### Mateusz Andrzejewski 

# %%

import os
import cv2
import time
import struct
import argparse
import numpy as np
from keras.models import Model
from keras.layers.merge import add, concatenate
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D

# %%
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%
arg_parser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

arg_parser.add_argument(
                        '-w',
                        '--weights',
                        help= 'path to weights file',
                        required=True)
arg_parser.add_argument(
                        '-o',
                        '--output',
                        help= 'output name. all output will be saved in outputs folder with this name',
                        required=False)    
group = arg_parser.add_mutually_exclusive_group(required=True)

group.add_argument(
                    '-i',
                    '--image',
                    help= 'path to image file')
                    
group.add_argument(
                    '-v',
                    '--video',
                    help= 'path to video file')


# %%
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance            

                    weights = norm_layer.set_weights([gamma, beta, mean, var])  

                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))     
    
    def reset(self):
        self.offset = 0

# %%
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objectiveness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objectiveness = objectiveness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

# %%
def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) - 2) and skip: skip_connection = x
        
        count += 1
        
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top

        x = Conv2D(conv['filter'], 
                   conv['kernel'], 
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']), 
                   use_bias=False if conv['bnorm'] else True)(x)

        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)

        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x]) if skip else x

# %%
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1: return 0
        return min(x2,x4) - x1

    if x2 < x3: return 0
    return min(x2,x4) - x3  

# %%
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# %%
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union

# %%
def make_yolo_v3_model():
    input_image = Input(shape=(None, None, 3))

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])    
    return model

# %%
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)/new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)
    return new_image

# %%
def decode_net_out(net_out, anchors, obj_thresh, nms_thresh, net_h, net_w):
    grid_h, grid_w = net_out.shape[:2]
    nb_box = 3
    net_out = net_out.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = net_out.shape[-1] - 5

    boxes = []

    net_out[..., :2]  = _sigmoid(net_out[..., :2])
    net_out[..., 4:]  = _sigmoid(net_out[..., 4:])
    net_out[..., 5:]  = net_out[..., 4][..., np.newaxis] * net_out[..., 5:]
    net_out[..., 5:] *= net_out[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = net_out[int(row)][int(col)][b][4]
            #objectness = net_out[..., :4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = net_out[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = net_out[int(row)][col][b][5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes

# %%
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

# %%
def do_nms(boxes, nms_thresh):
    
    if len(boxes) < 1: return boxes # if there are no boxes, just return

    nb_class = len(boxes[0].classes) # number of classes
        
    for c in range(nb_class): # for each class
        sorted_indices = np.argsort([-box.classes[c] for box in boxes]) # sort by class score

        for i in range(len(sorted_indices)): # for each box
            index_i = sorted_indices[i] # index of the i-th box

            if boxes[index_i].classes[c] == 0: continue # if the class is background, continue

            for j in range(i+1, len(sorted_indices)): # for all boxes that have the same class
                index_j = sorted_indices[j] # index of the j-th box

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh: # if the iou is larger than the threshold
                    boxes[index_j].classes[c] = 0 # set the class to background

# %%
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

# %%
def load_model(weights_path:str):
    yolo_v3 = make_yolo_v3_model()
    # load the weights trained on COCO into the model
    weight_reader = WeightReader(weights_path)
    weight_reader.load_weights(yolo_v3)
    return yolo_v3

# %%
def get_parameters():
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    class_threshold = 0.6
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    return net_h, net_w, obj_thresh, nms_thresh,class_threshold, anchors, labels    

# %%
def draw_boxes_on_photo(image, boxes, labels, obj_thresh, output_name:str):
    cv2.namedWindow("output", cv2.WINDOW_KEEPRATIO)
    COLORS = np.random.randint(50, 255, size=(len(labels), 3), dtype="uint8")

    #plot each box
    for i in range(len(boxes)):

        box = boxes[i] # get the ith box
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax  # get box coordinates
        width, height = x2 - x1, y2 - y1 # calculate width and height of the box
        color = [int(c) for c in COLORS[i]]
        cv2.rectangle(image,(x1,y1),(x1+width,y1+height),color,3)
        
        label = "%s (%.3f)" % (labels[i], obj_thresh[i])
        cv2.putText(image, label, (int(width*0.4), y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    cv2.imshow('output', image)
    cv2.imwrite(f"{output_name}.jpg", image)
    cv2.waitKey(0)

# %%
def analyze_image(image_path:str,output_name:str='new_image'):
    image = cv2.imread(image_path)
    if(image is None):
        print("Error opening photo! Please check the path and try again.") 
        return

    print("\n-----------------------------"  )
    print("\nstarting to analyze the photo\n")
    print(  "-----------------------------\n")

    timer = time.time()
    
    image_h, image_w, _ = image.shape
    net_h, net_w, obj_thresh, nms_thresh, class_threshold, anchors, labels = get_parameters()
    new_image = preprocess_input(image, net_h, net_w)
    
    # run the prediction
    yolo = yolo_v3.predict(new_image)
    boxes = []

    for i in range(len(yolo)):
        # decode the output of the network
        boxes += decode_net_out(yolo[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     
    
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # draw bounding boxes on the image using labels
    print(f"photo {output_name} finished.")
    print(f"\nTime taken to analyze the photo: {round((time.time() - timer), 3)}s.\n")
    draw_boxes_on_photo(image, v_boxes, v_labels, v_scores, output_name)

# %%
def analyze_video(video_path:str,output_name:str='new_video'):
    video = cv2.VideoCapture(video_path)
    if(video.isOpened() == False):
        print("Error opening video file! Please check the path and try again") 
        return

    print("\n-----------------------------"  )
    print("\nstarting to analyze the video\n")
    print(  "-----------------------------\n")

    count = 0
    net_h, net_w, obj_thresh, nms_thresh, class_threshold, anchors, labels = get_parameters()
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    frameCount   = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(f'{output_name}.mp4',cv2.VideoWriter_fourcc(*'mp4v'),24,(frame_width,frame_height))
    
    grabbed, frame = video.read()
    timer = time.time()

 #----------------------------------------------------------------------------------------------------------------------    

    while grabbed:

        frame_timer = time.time()
        count += 1
        boxes = []

        new_image = preprocess_input(frame, net_h, net_w)

        yolo = yolo_v3.predict(new_image)
        
        for i in range(len(yolo)):# decode the output of the network
            boxes += decode_net_out(yolo[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        
        correct_yolo_boxes(boxes, frame_height, frame_width, net_h, net_w)

        do_nms(boxes, nms_thresh)     
    
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        for i in range(len(v_boxes)): #plot each box

            box = v_boxes[i] # get the ith box
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax  # get box coordinates
            width, height = x2 - x1, y2 - y1 # calculate width and height of the box
            color = [int(c) for c in COLORS[i]]
            label = "%s (%.3f)" % (v_labels[i], v_scores[i])

            cv2.rectangle(frame,(x1,y1),(x1+width,y1+height), color,2)
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        writer.write(frame)

        print(f"frame processed: {count} of {frameCount} time taken: {round((time.time() - frame_timer), 3)}s")

        grabbed, frame = video.read()

 #----------------------------------------------------------------------------------------------------------------------
    video.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nVideo {output_name} processed.\n")
    print(f"sorry for taking so long!")
    print(f"Total time taken: {round((time.time() - timer), 3)}s.")


# %%
def main(args):
    print("Loading weights...")
    print("\n-----------------------------"  )

    global yolo_v3
    yolo_v3 = load_model(args.weights) # make the yolo v3 model to predict 80 classes on COCO
    
    print("\nweights loaded\n")
    
    if args.image != None:
        if args.output is not None:
            analyze_image(args.image, args.output)
        else: analyze_image(args.image)

    if args.video != None:
        if args.output is not None:
            analyze_video(args.video, args.output)
        else: analyze_video(args.video)

# %%
if __name__ == '__main__':
   args = arg_parser.parse_args() 
   main(args)