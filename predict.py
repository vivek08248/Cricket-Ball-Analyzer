from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)


model_path = os.path.join('runs','detect','train5','weights','best.pt')
model = YOLO(model_path)

video_path = os.path.join('videos','test3.mp4')
cap = cv2.VideoCapture(video_path)
ret = True
prevTime = 0
centroid_history = FixedSizeQueue(10)
start_time = time.time()
interval = 0.6
paused = False
angle = 0
prev_frame_time = 0 
new_frame_time = 0

while ret:
    ret, frame = cap.read()
    if ret:
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps)  
        fps = str(fps)
        print(list(centroid_history.queue))
        current_time = time.time()
        if current_time - start_time >= interval and len(centroid_history)>0:
            centroid_history.pop()
            start_time = current_time
        
        results = model.track(frame, persist=True,conf=0.35,verbose=False)
        boxes = results[0].boxes
        box = boxes.xyxy
        rows,cols = box.shape
        if len(box)!=0:
            for i in range(rows):
                x1,y1,x2,y2 = box[i]
                x1,y1,x2,y2 = x1.item(),y1.item(),x2.item(),y2.item()
                
                centroid_x = int((x1+x2)/2)
                centroid_y = int((y1+y2)/2)
                
                centroid_history.add((centroid_x, centroid_y))
                cv2.circle(frame,(centroid_x, centroid_y),radius=3,color=(0,0,255),thickness=-1)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            for i in range(1, len(centroid_history)):
                # if math.sqrt(y_diff**2+x_diff**2)<7:
                cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)    
                
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
            x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            if(x_diff!=0):
                m1 = y_diff/x_diff
                if m1==1:
                    angle = 90
                elif m1!=0:
                    angle = 90-angle_between_lines(m1)
                if angle>=45:
                        print("ball bounced")
            future_positions = [centroid_list[-1]]
            for i in range(1, 5):
                future_positions.append(
                    (
                        centroid_list[-1][0] + x_diff * i,
                        centroid_list[-1][1] + y_diff * i
                    )
                )
            print("Future Positions: ",future_positions)
            for i in range(1,len(future_positions)):
                cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
                cv2.circle(frame,future_positions[i],radius=3,color=(0,0,255),thickness=-1)
                

        text = "Angle: {:.2f} degrees".format(angle)
        cv2.putText(frame,text,(20,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2) 
        frame_resized = cv2.resize(frame, (1000, 600))
        cv2.imshow('frame',frame_resized)
         
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            paused = not paused
            
            while paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = not paused
                elif key == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()




# import numpy as np
# import cv2
# import time
# import os
# import math
# from collections import deque
# from ultralytics import YOLO


# def angle_between_lines(m1, m2=1):
#     if m1 != -1/m2:
#         angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
#         return angle
#     else:
#         return 90.0


# class FixedSizeQueue:
#     def __init__(self, max_size):
#         self.queue = deque(maxlen=max_size)
    
#     def add(self, item):
#         self.queue.append(item)
    
#     def pop(self):
#         self.queue.popleft()

#     def clear(self):
#         self.queue.clear()

#     def get_queue(self):
#         return self.queue
    
#     def __len__(self):
#         return len(self.queue)


# model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
# model = YOLO(model_path)

# video_path = os.path.join('videos', 'test1.mp4')
# cap = cv2.VideoCapture(video_path)
# ret = True
# prevTime = 0
# centroid_history = FixedSizeQueue(10)
# start_time = time.time()
# interval = 0.6
# paused = False
# angle = 0
# prev_frame_time = 0 
# new_frame_time = 0

# # Smoothing function for lines (Bezier curve)
# def create_bezier_curve(points, smoothness=50):
#     t = np.linspace(0, 1, smoothness)
#     curve = []
#     for i in range(smoothness):
#         x = (1 - t[i]) ** 2 * points[0][0] + 2 * (1 - t[i]) * t[i] * points[1][0] + t[i] ** 2 * points[2][0]
#         y = (1 - t[i]) ** 2 * points[0][1] + 2 * (1 - t[i]) * t[i] * points[1][1] + t[i] ** 2 * points[2][1]
#         curve.append([int(x), int(y)])
#     return np.array(curve, dtype=np.int32)


# while ret:
#     ret, frame = cap.read()
#     if ret:
#         new_frame_time = time.time() 
#         fps = 1/(new_frame_time-prev_frame_time) 
#         prev_frame_time = new_frame_time 
#         fps = int(fps)  
#         fps = str(fps)
        
#         current_time = time.time()
#         if current_time - start_time >= interval and len(centroid_history) > 0:
#             centroid_history.pop()
#             start_time = current_time
        
#         results = model.track(frame, persist=True, conf=0.35, verbose=False)
#         boxes = results[0].boxes
#         box = boxes.xyxy
#         rows, cols = box.shape
#         if len(box) != 0:
#             for i in range(rows):
#                 x1, y1, x2, y2 = box[i]
#                 x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
#                 centroid_x = int((x1 + x2) / 2)
#                 centroid_y = int((y1 + y2) / 2)
                
#                 centroid_history.add((centroid_x, centroid_y))
#                 cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
#         # Smoothly connecting centroids using Bezier curve
#         if len(centroid_history) > 2:
#             centroid_list = list(centroid_history.get_queue())
#             curve_points = []
#             for i in range(1, len(centroid_history)):
#                 mid_point = (
#                     int((centroid_list[i - 1][0] + centroid_list[i][0]) / 2),
#                     int((centroid_list[i - 1][1] + centroid_list[i][1]) / 2)
#                 )
#                 curve_points.append(mid_point)
            
#             bezier_curve = create_bezier_curve([centroid_list[0], curve_points[0], centroid_list[-1]])
#             cv2.polylines(frame, [bezier_curve], isClosed=False, color=(255, 0, 0), thickness=3)
            
#         # Calculate angle and future positions
#         if len(centroid_history) > 1:
#             centroid_list = list(centroid_history.get_queue())
#             x_diff = centroid_list[-1][0] - centroid_list[-2][0]
#             y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            
#             if x_diff != 0:
#                 m1 = y_diff / x_diff
#                 if m1 == 1:
#                     angle = 90
#                 elif m1 != 0:
#                     angle = 90 - angle_between_lines(m1)
            
#             future_positions = [centroid_list[-1]]
#             for i in range(1, 5):
#                 future_positions.append(
#                     (
#                         centroid_list[-1][0] + x_diff * i,
#                         centroid_list[-1][1] + y_diff * i
#                     )
#                 )
            
#             # Smoothly connect future positions
#             bezier_curve_future = create_bezier_curve([future_positions[0], future_positions[1], future_positions[-1]])
#             cv2.polylines(frame, [bezier_curve_future], isClosed=False, color=(0, 255, 0), thickness=3)
            
#             for pos in future_positions:
#                 cv2.circle(frame, pos, radius=3, color=(0, 0, 255), thickness=-1)
                
#         text = "Angle: {:.2f} degrees".format(angle)
#         cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
#         cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        
#         frame_resized = cv2.resize(frame, (1000, 600))
#         cv2.imshow('frame', frame_resized)
         
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF == ord(' '):
#             paused = not paused
            
#             while paused:
#                 key = cv2.waitKey(30) & 0xFF
#                 if key == ord(' '):
#                     paused = not paused
#                 elif key == ord('q'):
#                     break

# cap.release()
# cv2.destroyAllWindows()
# Testing Git commit
