from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = normalize_data(trainX,testX)

    model1 = load_model('models/CNN/CNN-0.8747000098228455-100epochs.keras')
    model2 = load_model('models/ResNet/ResNet-0.9111999869346619-50epochs.keras')
    model3 = load_model('models/DLA/DLA-0.910099983215332-50epochs.keras')
    
    probs_model1 = predict_probabilities(model1, testX)
    probs_model2 = predict_probabilities(model2, testX)
    probs_model3 = predict_probabilities(model3, testX)

    avg_probs = (probs_model1 + probs_model2 + probs_model3) / 3.0

    y_pred = np.argmax(avg_probs, axis=1)

    testY_categorical = np.argmax(testY, axis=1)

    accuracy = accuracy_score(testY_categorical, y_pred)
    print(f"Ensemble Accuracy: {accuracy}")
    
    conf_matrix = confusion_matrix(testY_categorical, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

def normalize_data(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

def predict_probabilities(model, X):
    return model.predict(X)

if __name__ == "__main__":
    main()