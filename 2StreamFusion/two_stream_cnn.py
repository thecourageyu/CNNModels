
# coding: utf-8

# In[ ]:


#!/opt/conda/bin/python


# In[ ]:


from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input


from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, MaxPooling3D
from keras.layers import concatenate, Input, merge, Flatten
from keras.layers.core import Reshape
from keras.optimizers import SGD, Adam
from keras import backend as K

from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.utils import multi_gpu_model 
from keras.utils import plot_model

import os
import time
import numpy as np

from data_generator import DataSet


# In[ ]:


print(K)
K.tf.__version__
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


def temporal_stream(input_shape=(224, 224, 20), verbose=1):
    opt_flow_conv = Sequential()
  
    #conv1
    opt_flow_conv.add(Conv2D(filters=96,  
                             kernel_size=(7, 7),  
                             strides=2, 
                             padding='same', 
                             data_format="channels_last",
                             input_shape=input_shape)) 
    
    opt_flow_conv.add(BatchNormalization())
    opt_flow_conv.add(Activation('relu'))
    opt_flow_conv.add(MaxPooling2D(pool_size=(2, 2)))

    #conv2
    opt_flow_conv.add(Conv2D(filters=256, 
                             kernel_size=(5, 5), 
                             strides=2, 
                             padding='same'))
    
    opt_flow_conv.add(Activation('relu'))
    opt_flow_conv.add(MaxPooling2D(pool_size=(2, 2)))

    #conv3
    opt_flow_conv.add(Conv2D(filters=512, 
                             kernel_size=(3, 3), 
                             strides=1, 
                             activation='relu', 
                             padding='same'))

    #conv4
    opt_flow_conv.add(Conv2D(filters=512, 
                             kernel_size=(3, 3), 
                             strides=1, 
                             activation='relu', 
                             padding='same'))

    #conv5
    opt_flow_conv.add(Conv2D(filters=512, 
                             kernel_size=(3, 3), 
                             strides=1, 
                             activation='relu', 
                             padding='same'))
    
    opt_flow_conv.add(Conv2D(filters=512, 
                             kernel_size=(5, 5), 
                             strides=1, 
                             activation='relu', 
                             padding='same'))
    
#     opt_flow_conv.add(MaxPooling2D(pool_size=(2, 2)))
    
    if verbose:
        print("opt_flow_conv input tensor = {0}".format(opt_flow_conv.input))
        print("opt_flow_conv output tensor = {0}".format(opt_flow_conv.output))
    
    return opt_flow_conv, opt_flow_conv.output

temporal_stream()


# In[ ]:


def spatial_stream(base_model="VGG16", 
                   input_tensor=Input(shape=(224, 224, 3), name='rgb_img'), 
                   include_top=False, 
                   remove_layers=1, 
                   trainable=1, 
                   verbose=1):
    
    if base_model == "VGG16":
        spatial_conv = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=include_top) 
    elif base_model == "ResNet50":
        spatial_conv = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=include_top) 
        
    for _ in range(remove_layers):
        spatial_conv.layers.pop()
    
    if trainable:
        for layer in spatial_conv.layers[0:]:
            layer.trainable = True
    else:
        for layer in spatial_conv.layers[0:]:
            layer.trainable = False
            
    # remove the last "remove_layers" layers 
    output_tensor = spatial_conv.layers[-1].output    
    
    if verbose:
        for i, layer in enumerate(spatial_conv.layers):
            print(i, layer.name)
            
        print("spatial_conv input tensor = {0}".format(spatial_conv.input))
        print("spatial_conv output tensor = {0}".format(output_tensor))
        

    return spatial_conv, output_tensor 

spatial_stream()


# In[ ]:


# temporal model
def temporal_cnn(img_shape, opt_flow_len, n_classes):
    
    img_width = img_shape[0]
    img_height = img_shape[1]
    
    rgb_img = Input(shape=(img_width, img_height, 3), name='rgb_img')

    spatial_conv, spatial_output = spatial_stream(input_tensor=rgb_img, remove_layers=1, verbose=0)
    print("spatial output tensor = ", spatial_output)

    temporal_conv, temporal_output= temporal_stream(input_shape=(img_width, img_height, opt_flow_len * 2), verbose=0)
    print("temporal output tensor = ", temporal_output)

    concat_2_stream = concatenate([spatial_output, temporal_output])
    print("concatenate tensor = ", concat_2_stream.shape)

    n_filters = int(concat_2_stream.shape[3] // 2)

    concat_2_stream = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1, padding='same', activation="relu")(concat_2_stream)
    print("conv2d = ", concat_2_stream)

    concat_2_stream = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(concat_2_stream)
    print("maxpool2d = ", concat_2_stream)

    x = Flatten()(concat_2_stream)
    x = Dense(4096)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation='softmax')(x)
    print(out)

    temporal_cnn = Model(inputs=[spatial_conv.input, temporal_conv.input], outputs=out)

    # merged = Merge([left_branch, middle_branch, right_branch], mode='concat')

    # final_model = Sequential()
    # final_model.add(merged)
    # final_model.add(Dense(10, activation='softmax'))

    # plot_model(model, to_file='demo.png', show_shapes=True)
    print(temporal_cnn.summary())
    return temporal_cnn

# n_frames_per_video = 5
# n_classes = 4
# img_shape = (64, 64)
# opt_flow_len = 10
# temporal_cnn(img_shape, opt_flow_len, n_classes)


# In[ ]:


def temporal_cnn_train(num_of_snip=1, 
                       opt_flow_len=10, 
                       image_shape=(224, 224),
                       batch_size=32, 
                       nb_epoch=100, 
                       saved_model=None,
                       class_limit=None, 
                       load_to_memory=False, 
                       name_str=None, 
                       gpus=1):
    
    # Get local time.
    time_str = time.strftime("%Y%m%d%H%M", time.localtime())

    if name_str == None:
        name_str = "temporal_cnn"
#         name_str = time_str

    # Callbacks: Save the model.
    saved_model_dir = os.path.join('out', 'checkpoints', name_str)
    
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
            
#     checkpointer = ModelCheckpoint(filepath=os.path.join(saved_model_dir, '{epoch:03d}-{val_loss:.3f}.hdf5'),
    checkpointer = ModelCheckpoint(filepath=os.path.join(saved_model_dir, 'temproal_cnn_model.hdf5'),
                                   verbose=1,
                                   save_best_only=True, 
                                   monitor="loss")

    # Callbacks: TensorBoard
    TensorBoard_dir = os.path.join('out', 'TB', name_str)
    
    if not os.path.exists(TensorBoard_dir):
            os.makedirs(TensorBoard_dir)
            
    tb = TensorBoard(log_dir=os.path.join(TensorBoard_dir))

    # Callbacks: Early stopper.
    early_stopper = EarlyStopping(monitor='loss', patience=10)

    # Callbacks: Save results.
    log_dir = os.path.join('out', 'logs', name_str)
    
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(log_dir, 'temporal_cnn_training-' + str(timestamp) + '.log'))

    # Callbacks: learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001)
    
    # Learning rate schedule.
#     lr_schedule = LearningRateScheduler(fixed_schedule, verbose=0)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(num_of_snip=num_of_snip,
                       opt_flow_len=opt_flow_len,
                       class_limit=class_limit)
    else:
        data = DataSet(num_of_snip=num_of_snip,
                       opt_flow_len=opt_flow_len,
                       image_shape=image_shape,
                       class_limit=class_limit)
        
#     print("temporal-train-, Show data list: {0}".format(data.data_list))
    
    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
#     steps_per_epoch = (len(data.data_list) * 0.7) // batch_size

    steps_per_epoch = batch_size
    
    if load_to_memory:
        # Get data.
        X, y = data.get_all_stacks_in_memory('train')
        X_test, y_test = data.get_all_stacks_in_memory('test')
    else:
        # Get generators.
        generator = data.stack_generator(batch_size, 'train')
        val_generator = data.stack_generator(batch_size, 'test', name_str=name_str)
        
    # Get the model.    
    # Replicates `model` on 4 GPUs. 
    # This assumes that your machine has 4 available GPUs. 
    nb_classes = len(data.classes)

    metrics = ['accuracy']
    if nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

    optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)

    
    if saved_model is not None:
        tmp_cnn = load_model(saved_model)
    else:
        
        if gpus > 1:
            print("# of GPUS:", gpus)

            tmp_model = temporal_cnn(img_shape=image_shape, opt_flow_len=opt_flow_len, n_classes=nb_classes)
            tmp_cnn = multi_gpu_model(tmp_model, gpus=gpus)
        else:
            print("# of GPUS:", gpus)

            tmp_model = temporal_cnn(img_shape=image_shape, opt_flow_len=opt_flow_len, n_classes=nb_classes)
            tmp_cnn = tmp_model

        tmp_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)


    
    # Fit!
    if load_to_memory:
        # Use standard fit.
#         temporal_cnn.model.fit(
        tmp_cnn.fit(X,
                    y,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[tb, early_stopper, csv_logger],
                    epochs=nb_epoch)
    else:
        # Use fit generator.
        print("temporal-train-, use fit generator")
#         temporal_cnn.model.fit_generator(
        tmp_cnn.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger, checkpointer, reduce_lr],
                validation_data=val_generator,
                validation_steps=1,
                max_queue_size=20,
                workers=1,
                use_multiprocessing=False)
        


# In[7]:


def main(gpus=2):
    """
    These are the main training settings. 
    Set each before running this file.
    """
    "=============================================================================="
    config = {"saved_model": os.path.join('out', 'checkpoints', 'temporal_cnn', 'temproal_cnn_model.hdf5')}

    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    # yuzhe 20180110
    num_of_snip = 1 # number of chunks(snippets) used for each video
    opt_flow_len = 10 # number of optical flow frames used
#     image_shape=(224, 224)
#     image_shape=(64, 64)

    image_shape=(32, 32)

    
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 128

    nb_epoch = 2222
#    nb_epoch = 3333
    name_str = None

    if os.path.exists(config["saved_model"]):
#         saved_model = load_model(config["saved_model"])
        saved_model = config["saved_model"]

        print("using saved model")
    
    temporal_cnn_train(num_of_snip=num_of_snip, 
                       opt_flow_len=opt_flow_len, 
                       saved_model=saved_model,
                       class_limit=class_limit, 
                       image_shape=image_shape,
                       load_to_memory=load_to_memory, 
                       batch_size=batch_size,
                       nb_epoch=nb_epoch, 
                       name_str=name_str,
                       gpus=gpus)
    


# In[ ]:


main(gpus=1)


# In[ ]:


metrics = ['accuracy']

# optimizer = SGD(lr=1e-2, momentum=0.9, nesterov=True)

optimizer = Adam()
temporal_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
# spatiotemporal_cnn.fit(x=[input1, input2], y=a, batch_size=1, epochs=1)


# In[ ]:


steps_per_epoch = 32
generator = gen_dataset.stack_generator(batch_size, 'train')

val_generator = gen_dataset.stack_generator(batch_size, 'test')

spatiotemporal_cnn.fit_generator(
                generator=generator,
                steps_per_epoch=steps_per_epoch,
                epochs=nb_epoch,
                verbose=1,
#                 callbacks=[tb, early_stopper, csv_logger, checkpointer, lr_schedule, reduce_lr],
                validation_data=val_generator,
                validation_steps=1,
                max_queue_size=20,
                workers=1,
                use_multiprocessing=False)


# In[ ]:


img_shape = (224, 224, 3)

num_of_snip = 5
# rgb_img = Input(shape=(224, 224, 3), name='rgb_img')
rgb_img = Input(shape=(224, 224, 3), name='rgb_img')
opt_flow = Input(shape=(224, 224, 20), name='opt_flow')

#===============================================================================
spatial_inputs = []
temporal_inputs = []
# multiple_inputs = []
output_tensors = []
for snip_idx in range(num_of_snip):
    spatial_conv, spatial_output = spatial_stream(input_tensor=Input(shape=(224, 224, 3), name='rgb_img_{0}'.format(snip_idx)), 
                                                  remove_layers=1, 
                                                  verbose=0)
    
    print("spatial output tensor = ", spatial_output)

    temporal_conv, temporal_output= temporal_stream(input_shape=(224, 224, 20), verbose=0)
    print("temporal output tensor = ", temporal_output)
    
    
    
#     multiple_inputs.append(spatial_conv.input)
#     multiple_inputs.append(temporal_conv.input)
    spatial_inputs.append(spatial_conv.input)
    temporal_inputs.append(temporal_conv.input)

    output_tensors.append(concatenate([spatial_output, temporal_output]))


spatial_inputs.append(spatial_conv_snip_5.input)
temporal_inputs.append(temporal_conv_snip_5.input)
output_tensors.append(concatenate([spatial_output_snip_5, temporal_output_snip_5]))

multiple_inputs = spatial_inputs + temporal_inputs
# print(multiple_inputs)
# print(output_tensors)


# concat_2_stream = K.stack(output_tensors, axis=3) # [Width, Height, Time, Channel]
concat_2_stream = K.stack(output_tensors, axis=1) # [?, Time, Width, Height, Channel]

# concat_2_stream = concatenate([spatial_output, temporal_output])
print("concatenate tensor = ", concat_2_stream.shape)

# concat_2_stream = Activation("relu")(concat_2_stream)
# print("concatenate tensor after activation = ", concat_2_stream.shape)

n_filters = int(concat_2_stream.shape[4] // 2)
print(type(n_filters))
# concat_2_stream = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=1, padding='same')(concat_2_stream)
# print("concatenate tensor after 1 x 1 convolution = ", concat_2_stream.shape)

Conv3D_2_stream = Conv3D(filters=n_filters, # 3d convolution over Time, Width, Height
                         kernel_size=(3, 3, 3), # specifying the depth (<< notice this), height and width of the 3D convolution window. 
                         strides=(1, 1, 1), 
#                          padding='valid', 
                         padding='same', 
                         data_format="channels_last", 
#                          dilation_rate=(1, 1, 1), 
                         activation="relu", 
                         use_bias=True)(concat_2_stream)

print("Conv3D output tensor" , Conv3D_2_stream)


MaxPool3D_2_stream = MaxPooling3D(pool_size=(num_of_snip, 2, 2), strides=(2, 2, 2), padding="valid")(Conv3D_2_stream)
#         pool3d = tf.layers.max_pooling3d(fusion_conv6, pool_size=[self.nFramesPerVid,2,2], strides=(2,2,2), padding='valid') # [?,1,7,7,512]  ?=batchsize 
print("MaxPool3D output tensor", MaxPool3D_2_stream)

# # MaxPool3D_2_stream = Reshape(target_shape=(13 * 13 * 512))(MaxPool3D_2_stream)
# # print("MaxPool3D output tensor after reshape", MaxPool3D_2_stream)


# x = Flatten(input_shape=MaxPool3D_2_stream.shape[1:])(MaxPool3D_2_stream)
x = Flatten()(MaxPool3D_2_stream)

print("Flattened tensor", x.shape)
# x = Flatten(concat_2_stream)

x = Dense(4096)(x)
# x = BatchNormalization()(x)
x = Activation('elu')(x)
x = Dropout(0.3)(x)
x = Dense(2048, activation='relu')(x)
print(x)
out = Dense(10, activation='softmax')(x)
print(out)

# model = Model(inputs=[rgb_img, opt_flow], outputs=out)
# spatiotemporal_cnn = Model(inputs=[spatial_conv.input, temporal_conv.input], outputs=out)
spatiotemporal_cnn = Model(inputs=multiple_inputs, outputs=out)

# spatiotemporal_cnn = Model(inputs=[rgb_img, opt_flow], outputs=out)


# merged = Merge([left_branch, middle_branch, right_branch], mode='concat')

# final_model = Sequential()
# final_model.add(merged)
# final_model.add(Dense(10, activation='softmax'))


# plot_model(model, to_file='demo.png', show_shapes=True)
print(spatiotemporal_cnn.summary())


# In[ ]:


num_of_snip=1
opt_flow_len=10 
saved_model=None
class_limit=None 
image_shape=(64, 64)
load_to_memory=False
batch_size=32
nb_epoch=100
name_str=None

gen_dataset = DataSet(num_of_snip=num_of_snip,
                      opt_flow_len=opt_flow_len,
                      image_shape=image_shape,
                      class_limit=class_limit)


generator = gen_dataset.stack_generator(batch_size, 'train')
val_generator = gen_dataset.stack_generator(batch_size, 'test')

# print(next(generator))
# tmp_list, tmp_cls = next(generator)

# print(tmp_list[0].shape)
# print(tmp_list[1].shape)
# print(tmp_cls.shape)
# print(tmp_cls)

# for _ in range(10000):
#     next(generator)


# In[ ]:


tmp_list, tmp_cls = next(generator)
print(len(tmp_list))
print(tmp_cls.shape)
# print(len(tmp_list[0]))
# print(tmp_list[0][batch_size - 1][0].shape)
# print(len(tmp_list[1]))
# print(tmp_list[1][batch_size - 1][0].shape)
# print(tmp_cls.shape)
# print(tmp_cls)


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

log_dir = os.path.join(os.getcwd(), 'out', 'logs', 'temporal_cnn')
for tmp in os.listdir(log_dir):
    if tmp[0] != ".":
        
        log_data = pd.read_csv(log_dir + "/" + tmp)
        
        print(log_data.columns)
        
        plt.figure(figsize=(10, 5))
        ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), rowspan=2, colspan=2)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        lg1 = ax1.plot(log_data["epoch"], log_data["loss"])
        lg2 = ax1.plot(log_data["epoch"], log_data["val_loss"], color="blue")
        ax1.legend()

        ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), rowspan=2, colspan=2)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("accuracy")
        lg2 = ax2.plot(log_data["epoch"], log_data["acc"])
        lg2 = ax2.plot(log_data["epoch"], log_data["val_acc"], color="blue")
        ax2.legend()
        
        plt.tight_layout()
        
# line, = ax.plot([1, 2, 3])
# ax.legend()


# In[ ]:


# prediction

saved_model = os.path.join(os.getcwd(), 'out', 'checkpoints', 'temporal_cnn', 'temproal_cnn_model_no_dropout_1.hdf5')
print(saved_model)

batch_size = 1000
class_limit = None  # int, can be 1-101 or None
# yuzhe 20180110
num_of_snip = 1 # number of chunks(snippets) used for each video
opt_flow_len = 10 # number of optical flow frames used
#     image_shape=(224, 224)
image_shape=(64, 64)

# model = temporal_cnn(img_shape, opt_flow_len, n_classes=5)


config={"etc": "./etc", 
        "data_list": "data_list_v1.csv", 
     "img_path": "/home/jovyan/at073-group20/20bn_jester_500/train",                          
#                          "img_path": "/home/jovyan/at073-group20/gesture2img", 
     "opt_flow_path": "/home/jovyan/at073-group20/20bn_jester_500/optflow/train"}
#                          "opt_flow_path": "/home/jovyan/at073-group20/gesture2flow_s224"})

gen_dataset = DataSet(num_of_snip=num_of_snip,
                      opt_flow_len=opt_flow_len,
                      image_shape=image_shape, config=config)

val_generator = gen_dataset.stack_generator(batch_size, 'test')

model = load_model(saved_model)

val_classes = gen_dataset.classes
print(val_classes)

one_hot_idx = [gen_dataset.get_class_one_hot(tmp) for tmp in val_classes]
print(one_hot_idx)



# pd.replace

y_true = []
y_pred = []
for predict_idx in range(1):
    X, y = next(val_generator)
    print(len(X))
    
    preds = model.predict(X)
    
    for tmp_idx in range(batch_size):
        y_true.append(val_classes[np.argmax(y[tmp_idx])])
        y_pred.append(val_classes[np.argmax(preds[tmp_idx])])
#         print(val_classes[np.argmax(y[tmp_idx])], val_classes[np.argmax(preds[tmp_idx])])
        


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use("ggplot")

get_ipython().run_line_magic('matplotlib', 'inline')
# Put the results in
# y_true = 
# y_pred = 


labels = sorted(list(set(y_true)))
cmx_d = confusion_matrix(y_pred, y_true, labels=labels)
cmxn_d = cmx_d.astype('float') / cmx_d.sum(axis=0)[np.newaxis ,:]
cmx_df = pd.DataFrame(cmx_d, index=labels, columns=labels)
cmxn_df = pd.DataFrame(cmxn_d, index=labels, columns=labels)

plt.figure(figsize = (14.5, 7))
plt.subplot(121)
sns.heatmap(cmxn_df, annot=True, cmap='YlGnBu', cbar=False)
plt.title('Normalized confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.subplot(122)
sns.heatmap(cmx_df, annot=True, cmap='YlGnBu', cbar=False)
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

