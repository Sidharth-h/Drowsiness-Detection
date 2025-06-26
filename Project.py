import cv2
import mediapipe as mp
import pygame
import threading
import math
import smtplib
from email.message import EmailMessage
import time
import numpy as np
from collections import deque   


#ear-> eye_aspect_ratio
lefteye = [362, 385, 387, 263, 373, 380]
righteye = [33, 160, 158, 133, 153, 144]
nose=[1,2,98,327,168]
mouth=[78,308,14,13,17,87,317]

indices_dict={"lefteye":lefteye,"righteye":righteye,"nose":nose,"mouth":mouth}

def create_analysis_overlay(landmarks,indices_dict,canvas_size=(400,400)):
    canvas=np.zeros((canvas_size[1],canvas_size[0],3),dtype=np.uint8)
    h,w=canvas_size
    zoom_factor=1.5
    offset_x,offset_y=w//8,h//8

    def draw_region(indices,color):
        points=[]
        for idx in indices:
            x=int(landmarks[idx].x*w*zoom_factor-offset_x)
            y=int(landmarks[idx].y*h*zoom_factor-offset_y)
            points.append((x,y))
            cv2.circle(canvas,(x,y),2,color,-1)
        if len(points)>1:
            cv2.polylines(canvas,[np.array(points,dtype=np.int32)],isClosed=True,color=color,thickness=1)

    draw_region(indices_dict["lefteye"],(0,255,0))
    draw_region(indices_dict["righteye"],(255,0,0))
    draw_region(indices_dict["nose"],(0,255,255))
    draw_region(indices_dict["mouth"],(255,0,255))

#Graph
    if len(ear_his)>1:
        graph_height=50
        graph_width=len(ear_his)
        graph_y_start=h-graph_width
        for i in range(1,graph_width):
            pt1=(i-1,int(graph_y_start+(1-ear_his[i-1])*graph_height))
            pt2=(i,int(graph_y_start+(1-ear_his[i])*graph_height))
            cv2.line(canvas,pt1,pt2,(0,255,255),1)
        cv2.putText(canvas, "EAR Graph", (5, graph_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return canvas

#Function
ear_his=deque(maxlen=100)
email_cool=20
last_email_time=0
current_time=time.time()
pygame.init()
pygame.mixer.init()
beep=pygame.mixer.Sound("D:/Intern/Open CV/Eye Detection/BEEP (Beep sound effect).mp3")

def send_email():
    msg= EmailMessage()
    msg['Subject']='Drowsiness Alert'
    msg['From']='sidharthh004@gmail.com'
    msg['To']='sidharthc777@gmail.com'
    msg.set_content('Drowsiness Detected! Please Check immediately')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
            smtp.login('sidharthh004@gmail.com','ghhn azrn rjgq vxkx')
            smtp.send_message(msg)
        print("Email Sent")

    except Exception as e:
        print("Error Sending Email",e)


def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def eye_aspect_ratio(landmarks, eye_indices):
    a = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    b = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    c = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    return (a + b) / (2.0 * c)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



cap = cv2.VideoCapture(1)

blink_count = 0
counter = 0
alarm_on = False
EAR_THRESHOLD = 0.25
DROWSY_Duration=3
drowsy_start=None

blank_canvas=np.zeros((400,400,3),dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            #Eye Framing
            h,w,_=frame.shape
            for idx in lefteye:
                x=int(landmarks[idx].x*w)
                y=int(landmarks[idx].y*h)
                cv2.circle(frame,(x,y),2,(0,255,0),-1)
            for idx in righteye:
                x=int(landmarks[idx].x*w)
                y=int(landmarks[idx].y*h)
                cv2.circle(frame,(x,y),2,(255,0,0),-1)

            leftear = eye_aspect_ratio(landmarks, lefteye)
            rightear = eye_aspect_ratio(landmarks, righteye)
            ear = (leftear + rightear) / 2.0

            ear_his.append(ear)
            if ear < EAR_THRESHOLD:
               if drowsy_start is None:
                    drowsy_start=time.time()
               elapsed=time.time()-drowsy_start

               if elapsed>=DROWSY_Duration:
                if not alarm_on:
                    alarm_on=True
                    beep.play(loops=-1)


                # cv2.putText(frame, "Drowsiness Alert!!!", (165, 60),
                #             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, "Drowsiness Alert!!!", (165 + 3, 60 + 3),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 3)
                cv2.putText(frame, "Drowsiness Alert!!!", (165, 60),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                #cv2.putText(frame,"Wake UP!!!!",(400,130),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                if time.time()-last_email_time>email_cool:
                    last_email_time= time.time()
                    threading.Thread(target=send_email).start()
            else:
                
                email_sent=False
                drowsy_start=None
                if alarm_on:
                    alarm_on = False
                    beep.stop()
                

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            analysis_overlay=create_analysis_overlay(landmarks,indices_dict)
    else:
        analysis_overlay=blank_canvas.copy()

    


#*****************************************************************************************
#Analysis Window
    analysis=255*np.ones((200,400,3),dtype=np.uint8)

    for i in range(1,len(ear_his)):
        x1=(i-1)*4
        x2=i*4
        y1=int(150-ear_his[i-1]*100)
        y2=int(150-ear_his[i]*100)
        cv2.line(analysis,(x1,y1),(x2,y2),(0,0,255),2)
        
    #Threshold Line
    threshold_y=int(150-EAR_THRESHOLD*100)
    cv2.line(analysis,(0,threshold_y),(400,threshold_y),(0,255,0),1)

    #Status
    status_text="ALERT!!" if alarm_on else "Normal"
    status_color=(0,0,255) if alarm_on else (0,200,0)
    cv2.putText(analysis,f"Status:{status_text}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,status_color,2)

    #Email Since
    since_email=int(time.time()-last_email_time)
    cv2.putText(analysis,f"Since Email:{since_email}s",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,50,50),1)

       
    cv2.imshow("Analyser",analysis_overlay)
    cv2.imshow("Drowsiness Detector", frame)
    


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
