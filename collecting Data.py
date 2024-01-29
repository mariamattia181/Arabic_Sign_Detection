import cv2
import os

Data = './data'

if not os.path.exists(Data):
    os.makedirs(Data)

classes = 2
data_size = 40

capture = cv2.VideoCapture(0)

for i in range(classes):
    if not os.path.exists(os.path.join(Data, str(i))):
        os.makedirs(os.path.join(Data, str(i)))
    print('{} collected'.format(i))

    while True:
        ret, frame = capture.read()
        cv2.putText(frame, "Press S to start", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('s'):
            break
    
    counter = 0
    while counter < data_size:
        ret, frame = capture.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(Data, str(i), '{}.jpg'.format(counter)), frame)
        counter += 1

capture.release()
cv2.destroyAllWindows()
