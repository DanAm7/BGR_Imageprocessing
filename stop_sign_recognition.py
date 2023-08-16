from itertools import cycle
import cv2
import numpy as np
from scipy.stats import itemfreq


hit_ratio = 0.7

is_detected = False



def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]


clicked = False                        
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv2.VideoCapture(0) 
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse) #Exit if mouse cliked


# Read and process frames in loop
success, frame = cameraCapture.read()



while success and not clicked: #Main loop

    cv2.waitKey(1)
            
    
    success, frame = cameraCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #turn fram into grayscale
    img = cv2.medianBlur(gray, 19)  #smooth filter apply on grey image
    cv2.imshow('test2', img)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40) #looking for circles in the img... return center point and radius


    if not circles is None:  
        circles = np.uint16(np.around(circles)) #from float to int
        
        
        # max_r, max_i = 0, 0
        # for i in range(len(circles[:, :, 2][0])): #scanning all circles detected
        #     if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
        #         max_i = i
        #         max_r = circles[:, :, 2][0][i]
                
        # x, y, r = circles[:, :, :][0][max_i]
        
        for O in range(len(circles[:, :, 2][0])):
            x, y, r = circles[:, :, :][0][O]
            
            if y > r and x > r:
                square = frame[int(y-r*0.9):int(y+r*0.9),int(x-r):int(x+r)] #cut square around circle
                
                dominant_color = get_dominant_color(square, 2) #Returns the most dominant color in the square in BGR
                print(dominant_color)
                cv2.imshow('test', square)
            
                red = dominant_color[2]
                green = dominant_color[1]
                blue = dominant_color[0]
                
                
                # Check if red is the most dominant color in the square by ratio of 30%          
                if ((blue < red*hit_ratio) and (green < red*hit_ratio)) and red > 110:
                    print("STOP")
                    is_detected = True
                
                # Check if the colors are very dark or very light
                elif (blue > 220) and (green > 220):
                    if red*0.95 > blue and red*0.95 > green:
                        print("STOP")
                        is_detected = True
                        
                elif red > 240:
                    if red*0.7 > blue and red*0.7 > green:
                        print("STOP")
                        is_detected = True


                if is_detected:
                    cv2.putText(frame, 'STOP', (int(x-(r/2)),y-r), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, 2)# Writing "STOP"
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  #green circle
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  #red dot
                    break

        
    is_detected = False
    cv2.imshow('camera', frame)


cv2.destroyAllWindows()
cameraCapture.release()
