#!/anaconda3/envs/tensorflow/python.exe

import numpy as np
import argparse
import cv2

def main(args):
    #video_input = cv2.Videovideo_inputture('slow.flv')
    video_input = cv2.VideoCapture(args.input_video)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    frame_grabbed, old_frame = video_input.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(frame_grabbed):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_grabbed,frame = video_input.read()
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            print()
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, color[i].tolist(),-1)
        try:
            img = cv2.add(frame,mask)
        except Exception:
            print()
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff


        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    video_input.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description=('This script tracks red object on the video'))
    parser.add_argument('-i',
            '--input_video',
            type=str,
            required=True,
            help='A video file that will be processed.')

    return parser.parse_args()

if __name__ == '__main__':
        main(parse_arguments())
