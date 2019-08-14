import cv2
import numpy as np

classifierFace = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
classifierEyes = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
camera = cv2.VideoCapture(1)

sample = 1
numberSamples = 25
id = input('Digit your id: ')
width, height = 220, 220
print("Capturing in faces....")
while True:
    connection, frame = camera.read()

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faceDetected = classifier.detectMultiScale(frameGray)
    faceDetected = classifierFace.detectMultiScale(frameGray, scaleFactor=1.5, minSize=(100, 100))

    for (x, y, l, a) in faceDetected:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        region = frame[y:y + a, x:x + l]
        regionGrayEye = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        eyesDetected = classifierEyes.detectMultiScale(regionGrayEye)

        for (ex, ey, el, ea) in eyesDetected:
            cv2.rectangle(region, (ex, ey), (ex + el, ey + ea), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(frameGray) > 110:
                    imageFace = cv2.resize(frameGray[y:y + a, x:x + l], (width, height))
                    cv2.imwrite("photos/people." + str(id) + "." + str(sample) + ".jpg", imageFace)
                    print("[Photo: " + str(sample) + " successfully captured]")
                    sample += 1

    cv2.imshow("Face", frame)
    cv2.waitKey(1)
    if sample >= numberSamples + 1:
        break

print("Successfully captured faces")
camera.release()
cv2.destroyAllWindows()
