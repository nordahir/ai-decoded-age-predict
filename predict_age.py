# Import required module
import cv2

def find_face(net, frame, conf_threshold=0.7):
    frame_open_cv_dnn = frame.copy()
    frame_height = frame_open_cv_dnn.shape[0]
    frame_width = frame_open_cv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_open_cv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_open_cv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame_open_cv_dnn, bboxes



ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Age Model
ageModel = "models/age_model/age_net.caffemodel"

# Network
ageProto = "models/age_model/age_deploy.prototxt"
ageNetwork = cv2.dnn.readNet(ageModel, ageProto)

padding = 10

def predict_age(frame):
    faceProto="models/face_model/opencv_face_detector.pbtxt"
    faceModel="models/face_model/opencv_face_detector_uint8.pb"


    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    faceNet=cv2.dnn.readNet(faceModel,faceProto)

    while cv2.waitKey(1)<0 :
        resultImg,faceBoxes=find_face(faceNet,frame)
        if not faceBoxes:
            print("No Facefound")
        else:
            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                ageNetwork.setInput(blob)
                agePreds=ageNetwork.forward()
                age=ageList[agePreds[0].argmax()]
                print("Age: ", f'{age[1:-1]}', " years")

                cv2.putText(resultImg, "Age: " f'{age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age", resultImg)

# Add the location of the image
photo = cv2.imread("images/nor_beard.jpg")
predict_age(photo)
