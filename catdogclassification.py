# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:52:47 2019

@author: Umair
"""

 import numpy as np
 import keras as k
 import matplotlib.pyplot as plt
 import sklearn as sk
 
 
 train_path = "E:\\LEARNING TARGET\\Resources\\Dataset\\images\\catdog\\Train"
 valid_path = "E:\\LEARNING TARGET\\Resources\\Dataset\\images\\catdog\\Valid"
 test_path  = "E:\\LEARNING TARGET\\Resources\\Dataset\\images\\catdog\\Test"

train_batches = k.preprocessing.image.ImageDataGenerator().flow_from_directory(
        train_path,target_size=(224,224),classes=["dog","cat"],batch_size=2) 

test_batches = k.preprocessing.image.ImageDataGenerator().flow_from_directory(
        test_path,target_size=(224,224),classes=['cat','dog'],batch_size=1) 

valid_batches = k.preprocessing.image.ImageDataGenerator().flow_from_directory(
        valid_path,target_size=(224,224),classes=['cat','dog'],batch_size=1)


def plots(ims=None, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


#it grabs total of 2 as batchsize is 2 from which some of them are cats others are dogs 
images , lables = next(train_batches)
plots(ims = images, titles = lables)

#noOfBatches = totalNumberOfSamples / batchSize
#NoOfBachesOfSamples to yield from generator before declaring one epcoh is finish

model = k.Sequential([
        k.layers.Conv2D(filters=32,kernel_size=(3,3), activation="relu", input_shape = (224,224,3)),
    k.layers.Flatten(),
    k.layers.Dense(2,activation="softmax")])

    
model.compile(k.optimizers.Adam(lr=0.0001),loss="categorical_crossentropy",metrics = ["accuracy"])

model.fit_generator(train_batches,steps_per_epoch=12/2,validation_data=valid_batches,
                    validation_steps=4/1,epochs=5,verbose=2)

images = []
lables = []
for i in np.arange(4):
    test_images , test_lables = next(test_batches)
    plots(ims= test_images ,titles=test_lables)    
    images.append(test_images)
    lables.append(test_lables)

images = np.array(images).reshape(-1,224,224,3)
lables = np.array(lables,dtype=np.float32).reshape(-1,2)
plots(ims=images,titles=lables)    

labs = lables[:,0]

predicted = model.predict_generator(test_batches,steps=4/1,verbose=0)

sk.metrics.confusion_matrix(labs,predicted[:,0])
#won 2014 imagenet competition   fined tuned model
#pre-trained model  includes all train weights
vgg16 = k.applications.vgg16.VGG16()

vgg16.model.summary()

 

