import sys

from ExerciseModelProcessor import ExerciseModelProcessor
from ExerciseModelTrainer import ExerciseModelTrainer
from ExerciseModelRunner import ExerciseModelRunner

## FV: Front View
## SV: Side View

models = {
    'deadlift_FV' : {'video': 'private/videos/deadlift_FV_updown.mp4',
                      'data': 'data/deadlift_FV_updown.csv',
                      'model': 'models/deadlift_FV_updown.pkl'},

    'squat_FV' : {'video': 'private/videos/squat_FV.mp4',
            'data': 'data/squat_FV.csv',
            'model': 'models/squat_FV.pkl'},

    'squat_SV' : {'video': 'private/videos/squat_SV.mp4',
            'data': 'data/squat_SV.csv',
            'model': 'models/squat_SV.pkl'},

    'benchpress_FV' : {'video': 'private/videos/benchpress_FV.mp4',
                'data': 'data/benchpress_FV.csv',
                'model': 'models/benchpress_FV.pkl'}
        }

def main():
    action = input("Enter the action to perform (process/train/run/quit): ").strip().lower()
    model_choice = input("Enter the model to use (deadlift_FV, squat_FV, squat_SV, benchpress_FV): ").strip()

    if model_choice not in models:
        print(f"Invalid model choice: {model_choice}")
        return

    if action == 'process':
        mainProcessor(model_choice)
    elif action == 'train':
        mainTrainer(model_choice)
    elif action == 'run':
        mainRunner(model_choice)
    elif action == 'quit':
        sys.exit()
    else:
        print(f"Invalid choice: {action}")

def mainProcessor(modelKey):
    modelValue = models[modelKey]
    processor = ExerciseModelProcessor(modelKey, modelValue['data'], modelValue['video'])
    processor.processVideo()

def mainTrainer(modelKey):
    modelValue = models[modelKey]
    trainer = ExerciseModelTrainer(modelValue['data'], modelValue['model'])
    trainer.constructPipeline()

def mainRunner(modelKey):
    modelValue = models[modelKey]
    runner = ExerciseModelRunner(modelKey, modelValue['model'])
    runner.run()

if __name__ == "__main__":
    main()