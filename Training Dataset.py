#%%
import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pickle
#%%
mp_hands=mp.solutions.hands #detecting el hand 
mp_drawing=mp.solutions.drawing_utils # btersem el hand landmarks
mp_drawing_styles=mp.solutions.drawing_styles #el colors eli btb2a 3ala el hand

#%%
hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

#%%
#3amalt download lel dataset
Dataset = "C:\\Users\\asus\\OneDrive\\Desktop\\Arabic Sign Detection\\archive (1)"

# 7atet el asami files eli feeha el sowar fe list esmaha category
CATEGORIES = [category for category in os.listdir(Dataset) if os.path.isdir(os.path.join(Dataset, category))]

print(len(CATEGORIES))

#%%
data=[] # array hansheel feeh el coordinates bta3et el dataset kolha
labels=[] #hansheel fee el categories bta3et el dataset

#%%
for dir_ in os.listdir(Dataset): #bn3ml loop 3ala el data kolha
    for img_path in os.listdir(os.path.join(Dataset,dir_)): #bnakhod el image path bta3 kol soora
        
        #image preprocessing:
        
        data_aux=[]
        image=cv2.imread(os.path.join(Dataset,dir_,img_path)) #bn3ml connect lel sora bel path bta3ha w taba3 anhi folder
        image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        input_shape=(224,224,3)
        image = cv2.resize(image, (input_shape[0], input_shape[1]))
        image = image / 255.0  # Normalize pixel values
        
        
        
        result=hands.process(image_RGB)
        
        if result.multi_hand_landmarks: #check law fe hand wla la 
            for landmarks in result.multi_hand_landmarks: #handakhod el landmarks w nrsemhaa
               for i in range(len(landmarks.landmark)): #hangeeb el coordinates bta3et koll landmark fel 2eed
                   x=landmarks.landmark[i].x
                   y=landmarks.landmark[i].y
                   
                   data_aux.append(x)
                   data_aux.append(y)
            
            data.append(data_aux)       
            labels.append(dir_)






#%%
print(len(data_aux))  

#%%
print(len(data))
#%%
print(len(labels))  
  
#%%


f=open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()        
    
    

        
        


