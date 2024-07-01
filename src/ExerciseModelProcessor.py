import cv2
import mediapipe as mp
import numpy as np
import csv

class ExerciseModelProcessor:
    def __init__(self, exerciseName, outputCSV, inputVideo):
        self.exerciseName = exerciseName,
        self.outputCSV = outputCSV
        self.inputVideo = inputVideo
        self.detectionConfidence = 0.5
        self.trackingConfidence = 0.5

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.landmarks = ['label']

        for i in range(1, 34):
            self.landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

        self.initCSV()

    def initCSV(self):
        with open(self.outputCSV, mode = 'a', newline = '') as f:
            csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            csv_writer.writerow(self.landmarks)

    def generateLabel(self, results, label):
        try:
            keypoints = self.generateKeywords(results, label)

            with open(self.outputCSV, mode = 'a', newline = '') as f:
                csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                csv_writer.writerow(keypoints)

        except Exception as e:
            print(f"Error generating landmarks: {e}")

    def generateKeywords(self, results, label):
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
            keypoints.append(landmark.visibility)

        keypoints = np.array(keypoints).flatten().tolist()
        keypoints.insert(0, label)
        return keypoints
    
    def processFrame(self, pose, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results

    def processVideo(self):
        cap = cv2.VideoCapture(self.inputVideo)

        with self.mp_pose.Pose(min_detection_confidence = self.detectionConfidence, 
                               min_tracking_confidence = self.trackingConfidence) as pose:
            while cap.isOpened():
                isReadable, frame = cap.read()

                if not isReadable:
                    break

                image, results = self.processFrame(pose, frame)

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                key = cv2.waitKey(1)

                if key == ord('w'):
                    self.generateLabel(results, 'up')
                elif key == ord('s'):
                    self.generateLabel(results, 'down')
                elif key == ord('d'):
                    self.generateLabel(results, 'right')
                elif key == ord('f'):
                    self.generateLabel(results, 'neutral')
                elif key == ord('g'):
                    self.generateLabel(results, 'left')

                cv2.imshow(f'{self.exerciseName} Feed', image)

                if cv2.waitKey(10) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()