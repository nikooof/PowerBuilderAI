import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk

class PowerBuilderGUI():
    def __init__(self):
        self.isUpdating = False
        self.model = None
        self.isSkeletonView = False
        self.videoCapture = cv2.VideoCapture(0)

        self.initMainWindow()
        self.initPoseDetection()
        self.initLandmarks()
        self.update()
        self.mainWindow.mainloop()

    def initMainWindow(self):
        self.mainWindow = tk.Tk()
        self.mainWindow.title("PowerBuilder AI")
        self.mainWindow.geometry("1000x550")

        self.mainWindowCanvas = tk.Canvas(self.mainWindow, width=1000, height=550)
        self.mainWindowCanvas.grid(row=0, column=0, rowspan=2, columnspan=2)

        for i in range(2):
            self.mainWindowCanvas.grid_columnconfigure(i, weight=1)

        self.drawGradient(self.mainWindowCanvas, "#141221", "#1e1d2f", 1000, 550, True)
        self.initVideoFrame()
        self.initControlFrame()

    def initVideoFrame(self):
        self.videoFrame = tk.Canvas(self.mainWindowCanvas, width=650, height=500, bg="white",
                                    highlightbackground='teal', highlightthickness=3)
        self.videoFrame.grid(row=0, column=1, padx=12.5, pady=25)
        self.videoFrameDefault()

    def videoFrameDefault(self):
        self.videoFrame.delete('all')
        if not self.isUpdating:
            self.videoFrame.create_text(325, 250, text="Welcome to Powerbuilder AI!",
                                        fil="#19A9A9", font=("Fira Code", 22))
            self.videoFrame.create_text(325, 275, text="Please select a model and press start to run the program!",
                                        fil="#19A9A9", font=("Fira Code", 18))
            self.videoFrame.configure(bg="#191929")

    def initControlFrame(self):
        self.controlsFrame = tk.Canvas(self.mainWindowCanvas, width=200, height=400, background='#141221',
                                       highlightbackground='teal', highlightthickness=3)
        self.controlsFrame.grid(row=0, column=0, padx=12.5, pady=25)

        for i in range(5):
            self.controlsFrame.grid_rowconfigure(i, weight=1)

        self.controlsFrame.grid_columnconfigure(0, weight=1)

        self.initOptionsFrame()
        self.initExerciseStatsFrame()
        self.initExerciseSelectionFrame()
        self.initToggleViewFrame()
        self.initQuitFrame()

    def initPoseDetection(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.repCount = 0
        self.movementPhase = ''
        self.updateDelay = 15
        self.pose = None

    def initLandmarks(self):
        self.landmarks = ['label']
        for i in range(1, 34):
            self.landmarks += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    def initOptionsFrame(self):
        self.optionsFrame = tk.Frame(self.controlsFrame, width=200, height=50, bg="black")
        self.optionsFrame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')

        self.optionsFrame.grid_columnconfigure(0, weight=1)
        self.optionsFrame.grid_columnconfigure(1, weight=1)

        startButton = tk.Button(self.optionsFrame, text="Start", command=self.startModel, highlightbackground="#141221",
                                fg='teal', font=("Fira Code", 12))
        startButton.grid(row=0, column=0, sticky='nsew')

        resetButton = tk.Button(self.optionsFrame, text="Reset", command=self.resetCounter,
                                highlightbackground="#141221", fg='teal', font=("Fira Code", 12))
        resetButton.grid(row=0, column=1, sticky='nsew')

    def initExerciseStatsFrame(self):
        self.exerciseStatsFrame = tk.Canvas(self.controlsFrame, width=200, height=75, bg='#141221',
                                            highlightbackground='#141221')
        self.exerciseStatsFrame.grid(row=1, column=0, padx=10, pady=5, sticky='news')

        self.exerciseStatsFrame.create_line(8.0, 66, 267.5, 66, fill="#19A9A9", width=2)

        self.exerciseStatsFrame.create_line(8.0, 4, 21.5, 4, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(254.0, 4, 267.5, 4, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(8, 3, 8, 16.5, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(267, 3, 267, 16.5, fill="#19A9A9", width=2)

        self.exerciseStatsFrame.create_line(8.0, 127.5, 21.5, 127.5, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(254.0, 127.5, 267.5, 127.5, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(8.5, 127.5, 8.5, 114, fill="#19A9A9", width=2)
        self.exerciseStatsFrame.create_line(267, 127.5, 267, 114, fill="#19A9A9", width=2)

        self.exerciseStatsFrame.grid_rowconfigure(0, weight=1)
        self.exerciseStatsFrame.grid_rowconfigure(1, weight=1)
        self.exerciseStatsFrame.grid_columnconfigure(0, weight=1)

        self.initMovementPhaseFrame()
        self.initRepCounterFrame()

    def initExerciseSelectionFrame(self):
        self.exerciseSelectionFrame = tk.Frame(self.controlsFrame, width=200, height=150, bg='#141221')
        self.exerciseSelectionFrame.grid(row=2, column=0, padx=10, pady=5, sticky='news')

        self.exerciseSelectionFrame.grid_rowconfigure(0, weight=1)
        self.exerciseSelectionFrame.grid_rowconfigure(1, weight=1)
        self.exerciseSelectionFrame.grid_rowconfigure(2, weight=1)
        self.exerciseSelectionFrame.grid_columnconfigure(0, weight=1)

        self.isSquatFrontView = tk.BooleanVar()
        self.isSquatSideView = tk.BooleanVar()
        self.isBenchFrontView = tk.BooleanVar()
        self.isDeadliftFrontView = tk.BooleanVar()

        squatFrame = tk.Canvas(self.exerciseSelectionFrame, width=200, height=150, bg='#141221',
                               highlightbackground='#19A9A9', highlightthickness=2)
        squatFrame.grid(row=0, column=0, padx=10, pady=5, sticky='news')

        squatLabel = tk.Label(squatFrame, height=2, width=10, text="Squat", fg="#19A9A9", bg="#141221", padx=3, pady=5,
                              anchor='w', font=("Fira Code", 12))
        squatLabel.grid(row=0, column=0, padx=5, pady=5)

        squatFrontViewButton = tk.Checkbutton(squatFrame, width=7, text='Front', variable=self.isSquatFrontView,
                                              bg='#141221', anchor='nw', command=self.squatFrontViewSelected, font=("Fira Code", 12))
        squatFrontViewButton.grid(row=0, column=1, sticky='w')
        squatSideViewButton = tk.Checkbutton(squatFrame, width=7, text='Side', variable=self.isSquatSideView,
                                             bg='#141221', anchor='nw', command=self.squatSideViewSelected, font=("Fira Code", 12))
        squatSideViewButton.grid(row=0, column=2, sticky='w', padx=10)

        benchPressFrame = tk.Canvas(self.exerciseSelectionFrame, width=200, height=150, bg='#141221',
                                    highlightbackground='#19A9A9', highlightthickness=2)
        benchPressFrame.grid(row=1, column=0, padx=10, pady=5, sticky='news')

        benchPressLabel = tk.Label(benchPressFrame, height=2, width=10, text="Bench Press", fg="#19A9A9", bg="#141221",
                                   padx=3, pady=5, anchor='w', font=("Fira Code", 12))
        benchPressLabel.grid(row=0, column=0, padx=5, pady=5)

        benchPressFrontViewButton = tk.Checkbutton(benchPressFrame, width=7, text='Front',
                                                   variable=self.isBenchFrontView, bg='#141221', anchor='nw',
                                                   command=self.benchPressFrontViewSelected, font=("Fira Code", 12))
        benchPressFrontViewButton.grid(row=0, column=1, sticky='w')

        deadliftFrame = tk.Canvas(self.exerciseSelectionFrame, width=200, height=150, bg='#141221',
                                  highlightbackground='#19A9A9', highlightthickness=2)
        deadliftFrame.grid(row=2, column=0, padx=10, pady=5, sticky='news')

        deadliftLabel = tk.Label(deadliftFrame, height=2, width=10, text="Deadlift", fg="#19A9A9", bg="#141221", padx=3,
                                 pady=5, anchor='w', font=("Fira Code", 12))
        deadliftLabel.grid(row=0, column=0, padx=5, pady=5)

        deadliftFrontViewButton = tk.Checkbutton(deadliftFrame, width=7, text='Front',
                                                 variable=self.isDeadliftFrontView, bg='#141221', anchor='nw',
                                                 command=self.deadliftFrontViewSelected)
        deadliftFrontViewButton.grid(row=0, column=1, sticky='w')

    def squatFrontViewSelected(self):
        if self.isSquatFrontView.get():
            self.deselectCheckboxes()
            self.isSquatFrontView.set(True)
            self.loadModel()
        else:
            self.deselectCheckboxes()

    def squatSideViewSelected(self):
        if self.isSquatSideView.get():
            self.deselectCheckboxes()
            self.isSquatSideView.set(True)
            self.loadModel()
        else:
            self.deselectCheckboxes()

    def benchPressFrontViewSelected(self):
        if self.isBenchFrontView.get():
            self.deselectCheckboxes()
            self.isBenchFrontView.set(True)
            self.loadModel()
        else:
            self.deselectCheckboxes()

    def deadliftFrontViewSelected(self):
        if self.isDeadliftFrontView.get():
            self.deselectCheckboxes()
            self.isDeadliftFrontView.set(True)
            self.loadModel()
        else:
            self.deselectCheckboxes()

    def initToggleViewFrame(self):
        self.toggleViewFrame = tk.Frame(self.controlsFrame, width=200, height=10, bg='white')
        self.toggleViewFrame.grid(row=3, column=0, padx=10, pady=0, sticky='news')

        self.toggleViewFrame.grid_rowconfigure(0, weight=1)
        self.toggleViewFrame.grid_columnconfigure(0, weight=1)

        self.toggleViewButton = tk.Button(self.toggleViewFrame, text = "Skeleton View", command=self.toggleView, highlightbackground="#141221",
                    fg='teal', font=("Fira Code", 12))
        self.toggleViewButton.grid(row=0, column=0, sticky='nsew')
    
    def toggleView(self):
        self.isSkeletonView = not self.isSkeletonView
        if self.isSkeletonView:
            self.toggleViewButton.configure(text="Normal View")
        else:
            self.toggleViewButton.configure(text="Skeleton View")

    def initQuitFrame(self):
        quitFrame = tk.Frame(self.controlsFrame, width=200, height=20, bg='white')
        quitFrame.grid(row=4, column=0, padx=10, pady=8, sticky='news')

        quitFrame.grid_rowconfigure(0, weight=1)
        quitFrame.grid_columnconfigure(0, weight=1)

        quitButton = tk.Button(quitFrame, text="Quit", command=self.mainWindow.quit, highlightbackground="#141221",
                               fg='teal', font=("Fira Code", 12))
        quitButton.grid(row=0, column=0, sticky='nsew')

    def initMovementPhaseFrame(self):
        movementPhaseFrame = tk.Canvas(self.exerciseStatsFrame, width=200, height=75, bg='#141221',
                                       highlightbackground="#141221")
        movementPhaseFrame.grid(row=0, column=0, padx=10, pady=5, sticky='n')

        movementPhaseLabel = tk.Label(movementPhaseFrame, height=2, width=12, text="Position", fg="#19A9A9",
                                      bg='#141221', padx=0, pady=5)
        movementPhaseLabel.grid(row=0, column=0, padx=5, pady=5)

        self.movementPhaseBox = tk.Label(movementPhaseFrame, height=2, width=12, text="N/A", fg="white", bg='#141221',
                                         padx=0, pady=5)
        self.movementPhaseBox.grid(row=0, column=1, padx=5, pady=5)

    def initRepCounterFrame(self):
        repCounterFrame = tk.Frame(self.exerciseStatsFrame, width=200, height=75, bg='#141221')
        repCounterFrame.grid(row=1, column=0, padx=10, pady=5, sticky='n')

        repCounterLabel = tk.Label(repCounterFrame, height=2, width=12, text="Reps", fg="#19A9A9", bg='#141221', padx=3,
                                   pady=5)
        repCounterLabel.grid(row=0, column=0, padx=5, pady=5)

        self.repCounterBox = tk.Label(repCounterFrame, height=2, width=12, text='0', fg="white", bg="#141221", padx=3,
                                      pady=5)
        self.repCounterBox.grid(row=0, column=1, padx=5, pady=5)

    def startModel(self):
        if self.model is None:
            return
        self.isUpdating = True
        self.update()

    def resetCounter(self):
        self.repCount = 0
        self.repCounterBox.configure(text=str(self.repCount))
        self.movementPhaseBox.configure(text="N/A")
        self.isUpdating = False
        self.deselectCheckboxes()
        self.videoFrameDefault()

    def loadModel(self):
        models = {
            "squat_FV": "models/squat_FV.pkl",
            "squat_SV": "models/squat_SV.pkl",
            "benchpress_FV": "models/benchpress_FV.pkl",
            "deadlift_FV": "models/deadlift_FV_updown.pkl"
        }

        self.model = None

        if self.isSquatFrontView.get():
            path = models["squat_FV"]
        elif self.isSquatSideView.get():
            path = models["squat_SV"]
        elif self.isBenchFrontView.get():
            path = models["benchpress_FV"]
        elif self.isDeadliftFrontView.get():
            path = models["deadlift_FV"]
        else:
            return

        with open(path, "rb") as f:
            self.model = pickle.load(f)

        if self.model:
            self.videoFrame.delete('all')
            self.videoFrame.create_text(325, 250, text="Model loaded successfully!",
                                        fill="#19A9A9", font=("Fira Code", 22))
            self.videoFrame.create_text(325, 275, text="Press Start to begin detection.",
                                        fill="#19A9A9", font=("Fira Code", 18))

    def deselectCheckboxes(self):
        self.isSquatFrontView.set(False)
        self.isSquatSideView.set(False)
        self.isBenchFrontView.set(False)
        self.isDeadliftFrontView.set(False)
        self.model = None
        self.videoFrameDefault()

    def update(self):
        if not self.isUpdating:
            return

        isReadable, frame = self.videoCapture.read()
        if isReadable:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (650, 500))

            if self.pose is None:
                self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            results = self.pose.process(frame)

            landmark_spec = self.mp_drawing.DrawingSpec(color=(25, 169, 169), thickness=2, circle_radius=2)
            connection_spec = self.mp_drawing.DrawingSpec(color=(26, 169, 169), thickness=2, circle_radius=2)

            if self.isSkeletonView:
                blankFrame = np.full(frame.shape, (25, 25, 41), dtype=np.uint8)
                self.mp_drawing.draw_landmarks(blankFrame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                               landmark_spec, connection_spec)
                displayedFrame = blankFrame
            else:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                displayedFrame = frame

            try:
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                results.pose_landmarks.landmark]).flatten().tolist()
                df = pd.DataFrame([row], columns=self.landmarks[1:])

                if self.model:
                    movementPhaseClass = self.model.predict(df)[0]
                    movementPhaseProbability = self.model.predict_proba(df)[0]

                    if movementPhaseClass == 'down' and movementPhaseProbability[
                        movementPhaseProbability.argmax()] >= .7:
                        self.movementPhase = 'Down'
                    elif self.movementPhase == 'Down' and movementPhaseClass == 'up' and movementPhaseProbability[
                        movementPhaseProbability.argmax()] >= .7:
                        self.movementPhase = 'Up'
                        self.repCount += 1

            except Exception as e:
                print(e)

            if self.isUpdating:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(displayedFrame))
                self.videoFrame.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.movementPhaseBox.configure(text=self.movementPhase)
                self.repCounterBox.configure(text=str(self.repCount))

        self.mainWindow.after(self.updateDelay, self.update)

    def drawGradient(self, canvas, start_color, end_color, width, height, isHorizontal):
        r1, g1, b1 = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
        r2, g2, b2 = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)

        if isHorizontal:
            for i in range(width):
                r = int(r1 + (r2 - r1) * (i / width))
                g = int(g1 + (g2 - g1) * (i / width))
                b = int(b1 + (b2 - b1) * (i / width))
                color = f'#{r:02x}{g:02x}{b:02x}'
                canvas.create_line(i, 0, i, height, fill=color)
        else:
            for i in range(height):
                r = int(r1 + (r2 - r1) * (i / height))
                g = int(g1 + (g2 - g1) * (i / height))
                b = int(b1 + (b2 - b1) * (i / height))
                color = f'#{r:02x}{g:02x}{b:02x}'
                canvas.create_line(0, i, width, i, fill=color)

    def __del__(self):
        if self.videoCapture.isOpened():
            self.videoCapture.release()


def main():
    PowerBuilderGUI()

if __name__ == '__main__':
    main()