import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    paths = [os.path.join('photos', f) for f in os.listdir('photos')]
    print(paths)
    faces = []
    ids = []

    for pathImage in paths:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(pathImage)[-1].split('.')[1])
        ids.append(id)
        faces.append(imageFace)

    return np.array(ids), faces

ids, faces = getImagemComId()
print("Treinando.....")
eigenface.train(faces, ids)
eigenface.write('classifierEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classifierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifierLBPH.yml')

print("Treinamento realizado com sucesso")