from pytube import YouTube
import cv2
import os

URL = 'https://www.youtube.com/watch?v=nclPaVQmCHc'

yt = YouTube(URL)
video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
video.download(filename='youtube_video.mp4')


# Comment all below code to download the video for testing

cap = cv2.VideoCapture('youtube_video.mp4')

if not os.path.exists('images'):
    os.makedirs('images')

count = 0
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join('images','ball_img_'+str(count)+'.jpg'),frame)
        count += 1
        frame_resized = cv2.resize(frame, (1000, 600))
        cv2.imshow('frame',frame_resized)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
os.remove('youtube_video.mp4')