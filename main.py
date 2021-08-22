import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import pywt  # wavelet Transform
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# img = cv2.imread('./test_images/sharapova1.jpg')
# print(img.shape)
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# plt.imshow(img, cmap='gray')
# plt.show()

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


# faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)  # detects face and give four values x, y, w, h
#
# # x, y, w, h = faces[0]  # if there is two face it store first face
# #
# # face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# # plt.imshow(face_img)
# # plt.show()
#
# cv2.destroyAllWindows()
# for (x, y, w, h) in faces:
#     face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = img_gray[y:y + h, x:x + w]
#     roi_color = face_img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
# plt.figure()
# plt.imshow(face_img, cmap='gray')
# plt.show()
#
# plt.imshow(roi_color, cmap='gray')  # roi- region of intrest
# plt.show()

# function to get a cropped face image if two eyes are detected

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


path_to_data = "./dataset/"  # to get data of images
path_to_cr_data = "./dataset/cropped/"  # to store a cropped images

# to store directry of data set

img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

# To save the cropped images in Cropeed folder

if os.path.exists(path_to_cr_data):  # if folder exists it will remove it and new one
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

# to crop the images from input dataset and store it in cropped folder inside that it has that clebrity name folder that folder has the own clebrity face images.
# storing cropped images dirct and celebrity_file_names_dict for model purpose

cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    print(celebrity_name)

    celebrity_file_names_dict[celebrity_name] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)

            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name

            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)  # name as key and images path as value
            count += 1


# Build Model
# We need to feature extraction from images data set useful in training nad testing model.

# Preprocessing: Use wavelet transform as a feature for traning our model
# In wavelet transformed image, you can see edges clearly and that can give us clues on various facial features such as eyes, nose, lips etc


def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H


# Class for images

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1

# Images in cropped folder can be used for model training. We will use these raw images along with wavelet transformed images to train our classifier.
# Let's prepare X and y now

X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

X = np.array(X).reshape(len(X), 4096).astype(float)  # to make train of x better
print(X.shape)

# Train Model

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), (
'svc', SVC(kernel='rbf', C=10))])  # from x train data we need first scalee and train so we using pipeline

pipe.fit(X_train, y_train)

score = pipe.score(X_test, y_test)
print(score)

print(classification_report(y_test, pipe.predict(X_test)))

# Let's use GridSearch to try out different models with different paramets. Goal is to come up with best modle with best fine tuned parameters

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    }
}

scores = []
best_estimators = {}
import pandas as pd

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

print(best_estimators)
print(best_estimators['svm'].score(X_test, y_test))
print(best_estimators['logistic_regression'].score(X_test, y_test))

best_clf = best_estimators['svm']

# Save the trained model

import joblib
# Save the model as a pickle in a file
joblib.dump(best_clf, 'saved_model.pkl')

# Save class dictionary
import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))
