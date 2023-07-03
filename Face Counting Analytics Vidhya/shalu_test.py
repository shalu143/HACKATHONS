95
import pandas as pd
import cv2
import os

from retinaface import RetinaFace



test=pd.read_csv('test_data.csv')

test_img=list(test["Name"])


#Detect faces using RetinaFace
face_match=[]
for i in test_img:
    jt = 'train_HNzkrPW (1)/image_data/'+i
    print(jt)
    img = cv2.imread(jt)
    img_faces = RetinaFace.detect_faces(img,threshold=0.95)
    try:
       z=len(img_faces.keys())
    except:
       z=0
    tf=[i,z]
    face_match.append(tf)


face_match_df=pd.DataFrame(face_match)
face_match_df.columns=["Name","HeadCount"]

face_match_df.to_csv('face_match_analytics_threshold_0.9.csv',index=False)

