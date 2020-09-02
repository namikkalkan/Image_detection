import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images,test_labels) = data.load_data()
train_images = train_images/255.0
test_images= test_images /255.0 

model = keras.Sequential ([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128 , activation = 'relu'),
    keras.layers.Dense (10, activation= 'softmax')
])

model.compile (optimizer = 'adam', loss = ('sparse_categorical_crossentropy'), metrics = ['accuracy'])
model.fit (train_images,train_labels,epochs =1)
test_acc,test_loss = model.evaluate (test_images,test_labels)
print (test_acc)
weights = model.get_weights()
print (np.shape(weights[1]))
print(np.shape(test_images[1]))
# print(np.shape(weights))
# 

prediction = model.predict(test_images)

for i in range (2):
    plt.grid (False)
    plt.imshow(test_images [i], cmap = plt.cm.binary)
    plt.xlabel('actual ' + class_names[test_labels[i]])
    plt.title('prediction '+ class_names[np.argmax(prediction[i])])
    plt.show()
    