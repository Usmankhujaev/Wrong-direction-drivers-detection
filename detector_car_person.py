# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import datetime as dt
import cv2
from tracker import Track, Tracker
import random as rand
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from collections import Counter
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": 'model/trained_weights_final_1749_2.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/car_class_2.txt',
        "score" : 0.3,
        "iou" : 0.40,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        h, w = image.size
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        centroids = []
        all_classes =[]
        lane_number = 0
        predicted_class = ""
       
        for i, c in reversed(list(enumerate(out_classes))):
            
            predicted_class = self.class_names[c]
            all_classes.append(predicted_class)
            box = out_boxes[i]
            top, left, bottom, right = box
            #y1, x1, y2, x2 
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(w, np.floor(bottom + 0.5).astype('int32'))
            right = min(h, np.floor(right + 0.5).astype('int32'))
         
            mid_x = int((left+right)/2)
            mid_y = int((bottom+top)/2)
            image = np.array(image)
            h, w, d= image.shape
            centroids.append(np.round(np.array([[mid_y], [mid_x]])))    
            
            
            #-------draw rectangles if car is detected--------------
            # uncomment these lines to the see the boxes
            #if c == 0:
                #cv2.rectangle(image, (left, top), (right, bottom), (255,255,0), 1)
                #cv2.putText(image, "Car", (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,0),1)
               
            if c == 1:
                cv2.rectangle(image, (left, top), (right, bottom), (0,0,255), 1)
                cv2.putText(image, "Person", (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),1)
                if (mid_y > h/3.86  and mid_y<=h/3.26):
                    lane_number = 1
                if (mid_y > h/2.42  and mid_y <=h/1.41):
                    lane_number = 2
                if (mid_y > h/1.41 and mid_y <=h):
                    lane_number = 3
                
                now = dt.datetime.now()
                raw_dir = './result/'+predicted_class+"_"
                lane_ = str(lane_number)+'_{0:02d}'.format(i)
                seconds =  now.second
                saving_name = "%s%s%s.jpg"%(raw_dir, now.strftime("%Y%m%d_%H%M%S_"), lane_)
                if os.path.exists(saving_name) == False and seconds%5 == 0:
                    if os.path.exists("./result") == False:
                        os.mkdir("./result")
                    print(saving_name)
                    resized = resize(image)
                    cv2.imwrite(saving_name,resized)
        
            try:
                image = Image.fromarray(image)
            except:
                print('no cars') 
              
        return image, centroids, out_scores, all_classes, out_boxes

    def close_session(self):
        self.sess.close()
def resize(img, scale=200):
    # percent of original size
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    tracker = Tracker(30,0,6,0)
    width = 400
    height = 300
    
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (50,50,98),(37,37,47)]
    pause = False
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps,(int(0.78*width)-int(0.109*width), height-int(height*0.0608)))
        
    fps = "FPS: ??"
    accum_time = 0
    curr_fps = 0
    clr = 0
   
    frame_counter = 0
    direction=""
    saving_name=""
    (dX, dY) = (0, 0)
    
    lane_number = 0
    centers = []
   
    prev_time = timer()
    while True:    
        return_value, frame = vid.read()
       
        if return_value is False:
            print('frame is empty: break')
            break
       
        orig_frame = frame.copy()
        frame = cv2.resize(frame, (width, height))
        frame = frame[int(0.0608*height):height, int(0.109*width):int(0.78*width)]
        h,w,d=frame.shape
       
        frame_counter+=1
           
        image = Image.fromarray(frame)
        new_result, centers, scores, out_classes, bbox = yolo.detect_image(image)
        

        result = np.array(new_result)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        
        if len(centers)>0 and len(out_classes)>0:
            tracker.Update(centers)
            for i in range(len(tracker.tracks)):
                x1=0
                y1=0
                x2=0
                y2=0
                if(len(tracker.tracks[i].trace)>1): 
                    for j in range(len(tracker.tracks[i].trace)-1):
                        y1 = tracker.tracks[i].trace[j][0][0]
                        x1 = tracker.tracks[i].trace[j][1][0]
                        y2 = tracker.tracks[i].trace[j+1][0][0]
                        x2 = tracker.tracks[i].trace[j+1][1][0]
                        clr = tracker.tracks[i].track_id
                        try:
                            some_position = tracker.tracks[i].trace[2][1][0]
                            some_position_y = tracker.tracks[i].trace[2][0][0]

                        except:
                            some_position = tracker.tracks[i].trace[1][1][0]
                            some_position_y = tracker.tracks[i].trace[1][0][0]
               
                        #cv2.circle(result, (int(x2), int(y2)), 2, track_colors[clr%3], -1)
                        cv2.arrowedLine(result, (int(some_position), int(some_position_y)), (int(x2), int(y2)),track_colors[clr%3], line_type=cv2.LINE_AA, thickness=1)

                        dX = x2-some_position
                        #dXCheck = x2-x1
                        dY = y2-y1
                    (dirX, dirY) = ("", "")
                    #print(dX)
                    if np.abs(dX)>=12:
                        dirX = "East" if np.sign(dX) == 1 else "West"
                    #if np.abs(dY)>=4:
                    #    dirY = "North" if np.sign(dY) == 1 else "South"
                   
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)
                    else:
                        direction = dirX if dirX != "" else dirY

                    if direction == "East":
                        if (y2 > 0 and y2<=h/2.28):
                            lane_number = 1
                        if (y2 > h/3.68  and y2 <=h/1.35):
                            lane_number = 2
                        if (y2 > h/1.35 and y2 <=h):
                            lane_number = 3
                       
                        cv2.putText(result, 'WRONG DIRECTION', (int(x1)-25, int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)
                        now = dt.datetime.now()
                        raw_dir = './result/wrong_direction_'
                        lane_ = str(lane_number)+'_{0:02d}'.format(i)
                        seconds =  now.second
                        right_now =  now.strftime("%Y%m%d_%H%M%S_") 
                        saving_name = "%s%s%s.jpg"%(raw_dir,right_now,lane_)
                        
                        if os.path.exists(saving_name) == False:
                            if os.path.exists("./result") == False:
                                os.mkdir("./result")
                            print(saving_name)
                            resized = resize(result)
                            cv2.imwrite(saving_name,resized)
                            
                    else:
                        cv2.putText(result, direction, (int(x1)-25, int(y1)-25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1)
                    
       
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
      
        cv2.putText(result, "Count: "+str(len(centers)), (3, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1 )        
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.40, color=(0, 255, 0), thickness=1)
       
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        k = cv2.waitKey(1) & 0xff
        if isOutput:
            try:
                out.write(result)
            except: continue
        if k == ord('q'):
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break
    yolo.close_session()
