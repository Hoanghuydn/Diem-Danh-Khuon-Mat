
'''########### Making single predictions ###########'''
import numpy as np
from keras.preprocessing import image
from lenet1 import classifier
from maping import ResultMap

ImagePath="datasets/0/User.0.1.jpg"
test_image=image.load_img(ImagePath,target_size=(64, 64))
test_image=image.img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=classifier.predict(test_image,verbose=0)
#print(training_set.class_indices)
 
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])