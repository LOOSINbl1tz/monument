import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_metrics(confusion_matrix):
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1_score


model_path = os.path.join('model','best','resnet50_2023-10-23-39-27.h5')
model = load_model(model_path)

test_data_dir = 'data/test'
test_generator = ImageDataGenerator().flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_labels = test_generator.classes
print(true_labels)

cm = confusion_matrix(true_labels, predicted_classes)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
# plt.axis('off')
plt.show()

print("Classification Report:")
print(classification_report(true_labels, predicted_classes))

accuracy, recall, precision, f1_score = calculate_metrics(cm)

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")