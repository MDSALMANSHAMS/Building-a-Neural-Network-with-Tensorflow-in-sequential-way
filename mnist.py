import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Input,Dense,MaxPool2D,BatchNormalization,GlobalAvgPool2D
from tensorflow.python.keras import activations
import matplotlib.pyplot as plt     # Flatten layer will also give the same output GlobalAvgPool2D, which is in array form                  
import numpy as np                  # but the difference between them is GlobalAvgPool2D will give much smaller size array.


# tensorflow.keras.Sequential
model=tf.keras.Sequential(                       # 2 dim. convolutional need 4 dim. tensor
    [                                            # Image input should be in 4 dim., i.e, (60000,28,28,1) (no.of image,height,width,color)
        Input(shape=(28,28,1)),                  # inpute the image, we can't change the input and the ouput (last Dense layer)
        Conv2D(32,(3,3),activation='relu'),      # input and output should be according to the dataset (image & labels)
        Conv2D(64,(3,3),activation='relu'),      # rest of the layers (Conv2D,MaxPool2D,BatchNormalization) we can change,
        MaxPool2D(),                             # as they are hyperparameters, so we can tune it accordingly
        BatchNormalization(),

        
        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),                       # this will flatten the image in the array form 
        Dense(64,activation='relu'),
        Dense(10,activation='softmax')           # this dense layer will give 10 probability for 10 different classes
                                                 # in mnist, we have 10 different classes 

    ]
)

def display_images(examples,labels):    
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx=np.random.randint(0,examples.shape[0]-1)
        img=examples[idx]
        label=labels[idx]

        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img,cmap='gray')

    plt.show()


if __name__=='__main__':
    
    (xtrain,ytrain),(xtest,ytest)=tf.keras.datasets.mnist.load_data()
    
    # x=image & y=label 
    print('xtrain.shape',xtrain.shape)
    print('ytrain.shape',ytrain.shape)
    print('xtest.shape',xtest.shape)
    print('ytest.shape',ytest.shape)

    if False:
        display_images(xtrain,ytrain)


    # Normalization
    xtrain=xtrain.astype('float32')/255
    xtest=xtest.astype('float32')/255

    # Expanding the dimension of the image, (as convolutional layers input the image in 4 dim., & we have the image of 3 dim..)
    xtrain=np.expand_dims(xtrain,axis=-1)  #(Image=(600000,28,28), Convolutional layers requirment=(600000,28,28,1))
    xtest=np.expand_dims(xtest,axis=-1)


    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

    # model training
    model.fit(xtrain,ytrain,batch_size=64,epochs=3,validation_split=0.2)

    # Evaluation on test set
    model.evaluate(xtest,ytest,batch_size=64)



