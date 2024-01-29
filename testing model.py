#%%
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

#%%
#SVC
model_dict=pickle.load(open('./modelSVC.pickle','rb'))
model=model_dict['modelSVC']





#%%
cap=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands 
mp_drawing=mp.solutions.drawing_utils 
mp_drawing_styles=mp.solutions.drawing_styles
    
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)






#%%

while True:
    
    data_aux=[]
    x_=[]
    y_=[]
    ret, frame=cap.read()
    
    H, W, _ = frame.shape
    frame_RGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
    result=hands.process(frame_RGB)
        
    if result.multi_hand_landmarks: 
        for landmarks in result.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                frame,  # image to draw
                landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        if result.multi_hand_landmarks: 
            for landmarks in result.multi_hand_landmarks: 
               for i in range(len(landmarks.landmark)): 
                   x=landmarks.landmark[i].x
                   y=landmarks.landmark[i].y
                   
                   data_aux.append(x)
                   data_aux.append(y)
                   x_.append(x)
                   y_.append(y)
            
            
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)
            
            
            
            
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            print(predicted_character)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)  
    
      
    cv2.imshow('frame',frame)
    key= cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    

    
cv2.release()
cv2.destroyAllWindows()

# %%
