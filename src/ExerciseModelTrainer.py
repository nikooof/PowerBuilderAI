import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score

class ExerciseModelTrainer:
    def __init__(self, dataPath, modelPath):
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.df = pd.read_csv(dataPath)
        self.x = self.df.drop('label', axis = 1)
        self.y = self.df['label']
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y, test_size = 0.25, random_state = 10)
        self.parameterGrid = {
            'gradientboostingclassifier__n_estimators': [50, 100, 200],
            'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
            'gradientboostingclassifier__max_depth': [3, 5, 7],
        }

    def constructPipeline(self):
        self.trainingPipeline = make_pipeline(StandardScaler(), GradientBoostingClassifier())
        gridSearch = GridSearchCV(estimator=self.trainingPipeline, param_grid=self.parameterGrid, 
                                  cv=5, n_jobs=-1, scoring='precision_macro')
        self.model = gridSearch.fit(self.xTrain, self.yTrain).best_estimator_
        with open(self.modelPath, 'wb') as f:
            pickle.dump(self.model, f)

    def evaluateModel(self):
        yPreds = self.model.predict(self.xTest)
        return precision_score(self.yTest, yPreds, average='macro', labels=['up', 'down'])