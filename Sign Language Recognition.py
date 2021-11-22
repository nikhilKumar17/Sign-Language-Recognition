#!/usr/bin/env python
# coding: utf-8

# ## 1. Import and Install Dependencies

# In[1]:


get_ipython().system('pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib')


# In[2]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# ## 2.Keypoints using MP Holistic

# In[3]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[4]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# In[5]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections 


# In[7]:


mp_holistic.POSE_CONNECTIONS # mp_drawing.draw_landmarks??


# In[125]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
         # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[7]:


results.face_landmarks 


# In[8]:


len(results.left_hand_landmarks.landmark)


# In[9]:


results


# In[10]:


frame  


# In[11]:


plt.imshow(frame)


# In[12]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[13]:


results


# In[14]:


draw_landmarks(frame, results)


# In[15]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[126]:



cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
   while cap.isOpened():

       # Read feed
       ret, frame = cap.read()
        # Make detections
       image, results = mediapipe_detection(frame, holistic)
       print(results)
       
#          Draw landmarks# after this be able to see our landmarks drawn to the screen in real time 
       draw_landmarks(image, results)
       # Show to screen
       cv2.imshow('OpenCV Feed', image)

       # Break gracefully
       if cv2.waitKey(10) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[17]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )  


# In[136]:


# or 
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )  


# In[142]:



cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
   while cap.isOpened():

       # Read feed
       ret, frame = cap.read()
        # Make detections
       image, results = mediapipe_detection(frame, holistic)
       print(results)
       
#          Draw landmarks 
       draw_styled_landmarks(image, results)
       # Show to screen
       cv2.imshow('OpenCV Feed', image)

       # Break gracefully
       if cv2.waitKey(10) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# ## 3. Extract Keypoint Values

# In[41]:


results


# In[42]:


results.pose_landmarks.landmark[0]# grabbing our first value


# In[43]:


results.pose_landmarks.landmark[0].x # grabbing our x


# In[44]:


results.pose_landmarks.landmark[0].y # grabbing our y


# In[45]:


for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
#     test.append(test)


# In[46]:


test # that is one set for landmarks


# In[47]:


# after result upto test 
# update will on below


# In[48]:


len(results.pose_landmarks.landmark)


# In[49]:


pose =[]
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)


# In[50]:


pose 
# there's 33 value


# In[51]:


len(pose)
# gives us the ability to work with each of these landmarks


# In[52]:


pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) 
 


# In[53]:


pose


# In[54]:


np.shape(pose)


# In[55]:


pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
 


# In[56]:


np.shape(pose)


# In[57]:


lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()


# In[58]:


lh


# In[59]:


lh.shape


# In[60]:


rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
# This error occurred because we didn't show your right hand in the video capture


# In[61]:


np.zeros(21*3) # 21 landmarks by 3 coordinate valules each
 
# create a blank array that look a little bit like that


# In[62]:


np.zeros(21*3).shape # simliar like left hand


# In[63]:


pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
# just add an if statement to the end of flatten to return a blank array if we don't actually have any value


# In[64]:


rh.shape


# In[65]:


rh


# In[66]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# In[67]:


extract_keypoints(results).shape


# In[68]:


result_test = extract_keypoints(results)


# In[69]:


result_test


# In[70]:


468*3+33*4+21*3+21*3 
#landmarks*3 coordinate


# In[71]:


np.save('0', result_test)


# In[72]:


np.load('0.npy')


# ## 4. Setup Folders for Collection

# In[73]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


# In[74]:


# hello
## 0
## 1
## 2
## ...
## 29
# thanks

# I love you


# In[75]:


for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# ## 5. Collect Keypoint Values for Training and Testing¶

# In[77]:


cap = cv2.VideoCapture(0 )
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
 
    # NEW LOOP 
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
# 30 frames if keypoints per video


                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()


# ## 6. Preprocess Data and Create Labels and Features

# In[78]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[79]:


label_map = {label:num for num, label in enumerate(actions)}


# In[80]:


label_map


# In[81]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[82]:


np.array(sequences).shape


# In[83]:


np.array(labels).shape


# In[84]:


X = np.array(sequences)


# In[85]:


X.shape


# In[86]:


y = to_categorical(labels).astype(int)


# In[87]:


y


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[89]:


y_test.shape


# ## 7. Build and Train LSTM Neural Network¶

# In[90]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[91]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[92]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[93]:


actions.shape[0]


# In[94]:


res = [.7, 0.2, 0.1]


# In[95]:


actions[np.argmax(res)]


# In[96]:


res[np.argmax(res)]


# In[97]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[98]:


model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])


# In[99]:


model.summary()


# ## 8. Make Predictions

# In[100]:


model.predict(X_test)


# In[101]:


res = model.predict(X_test)


# In[102]:


np.sum(res[0])


# In[103]:



actions[np.argmax(res[4])]


# In[104]:


actions[np.argmax(res[0])]


# In[105]:


actions[y_test[0]]


# In[106]:


actions[np.argmax(y_test[4])]


# ## 9. Save Weights

# In[107]:


model.save('action.h5')


# In[108]:


# del model


# In[109]:


model.load_weights('action.h5')


# ## 10. Evaluation using Confusion Matrix and Accuracy

# In[110]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[111]:


yhat = model.predict(X_test)


# In[112]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[113]:


multilabel_confusion_matrix(ytrue, yhat)


# In[114]:


accuracy_score(ytrue, yhat)


# ## 11. Test in Real Time

# In[117]:


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[118]:


plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))


# In[ ]:


sequence.reverse()


# In[126]:


len(sequence)


# In[ ]:


sequence.append('def')


# In[ ]:


sequence.reverse()


# In[127]:


sequence[-30:]


# In[139]:


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
#             image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[91]:


X_test[0]


# In[ ]:


#   model.predict(X_test[0]) 


# In[92]:


X_test[0].shape


# In[93]:


np.expand_dims(X_test[0],axis=0)


# In[141]:


# res[np.argmax(res)]>threshold


# In[95]:


np.argmax(res)


# In[94]:


model.predict((np.expand_dims(X_test[0],axis=0)))


# In[57]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:




