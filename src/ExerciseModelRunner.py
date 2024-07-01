import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np

class ExerciseModelRunner:
    def __init__(self, exerciseName, modelPath):
        self.exerciseName = exerciseName
        self.modelPath = modelPath
        self.inputVideo = 0

        self.detectionConfidence = 0.5
        self.trackingConfidence = 0.5

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.landmarks = ['label']

        for i in range(1, 34):
            self.landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    def run(self):
        cap = cv2.VideoCapture(self.inputVideo)
        
        repCount = 0
        movementPhase = ''

        with open(self.modelPath, 'rb') as f:
            self.modelOne = pickle.load(f)

        with self.mp_pose.Pose(min_detection_confidence = self.detectionConfidence, 
                               min_tracking_confidence = self.trackingConfidence) as pose:

            while cap.isOpened():
                isReadable, frame = cap.read()

                if not isReadable:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                try: 
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                    df = pd.DataFrame([row], columns= self.landmarks[1:])

                    movementPhaseClass = self.modelOne.predict(df)[0]
                    movementPhaseProbability = self.modelOne.predict_proba(df)[0]

                    if movementPhaseClass == 'down' and movementPhaseProbability[movementPhaseProbability.argmax()] >= .7:
                        movementPhase = 'Down'
                    elif movementPhase == 'Down' and movementPhaseClass == 'up' and movementPhaseProbability[movementPhaseProbability.argmax()] >= .7:
                        movementPhase = 'Up'
                        repCount += 1
                    
                    topLeft, btmRight, color, thickness = (0, 0), (220, 60), (0, 0, 0), 2
                    movementPhasePosition, repsPos = (15, 20), (15, 40)

                    textProperties = {
                        "fontFace": cv2.FONT_HERSHEY_COMPLEX,
                        "fontScale": 0.5,
                        "color": (255, 255, 255),
                        "thickness": 1,
                        "lineType": cv2.LINE_AA
                    }

                    cv2.rectangle(image, topLeft, btmRight, (0, 0, 255), thickness)
                    cv2.rectangle(image, topLeft, btmRight, color, -1)
                    cv2.putText(image, f"Movement Phase: {movementPhase}", movementPhasePosition, **textProperties)
                    cv2.putText(image, f"Rep count: {repCount}", repsPos, **textProperties)

                except Exception as e:
                    print(e)

                cv2.imshow(f'{self.exerciseName} Feed', image)

                if cv2.waitKey(10) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
