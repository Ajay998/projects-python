from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
model = load_model('/home/flock/PycharmProjects/projects-python/machine_learning/malaria-detection/a95e30model.h5')
model.summary()

test_image = image.load_img(
    '/home/flock/PycharmProjects/projects-python/machine_learning/malaria-detection/archive/cell_images/cell_images/Parasitized/C33P1thinF_IMG_20150619_115808a_cell_205.png', target_size=(50,50,3))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
print(result)
cell = "PARASITIZED" if result[0][1]==1 else "UNINFECTED"
print(cell)