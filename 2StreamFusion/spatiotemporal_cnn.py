#!/opt/conda/bin/python
# coding: utf-8

# In[ ]:




# In[2]:


from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input


from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, BatchNormalization, Conv2D, Conv3D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, MaxPooling3D
from keras.layers import Lambda, concatenate, Input, merge, Flatten, Reshape#, Concatenate
# from keras.layers.core import Reshape
from keras.optimizers import SGD, Adam
from keras import backend as K

from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.utils import multi_gpu_model 
from keras.utils import plot_model

import os
import time
import numpy as np

from data_generator import DataSet


# In[2]:


print(K)
K.tf.__version__
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# In[6]:


def temporal_stream(input_shape=(224, 224, 20), verbose=1):
        
    input_tensor = Input(shape=input_shape)
    
    # conv1
    opt_flow_conv = Conv2D(filters=96, 
                           kernel_size=(7, 7),
                           strides=(2, 2), 
                           padding="same",
                           data_format="channels_last",
                           input_shape=input_shape,
                           name="vggm_block1_conv1")(input_tensor)

    opt_flow_conv = BatchNormalization()(opt_flow_conv)
    opt_flow_conv = Activation("relu")(opt_flow_conv)
    opt_flow_conv = MaxPooling2D(pool_size=(2, 2), name="vggm_block1_maxpool")(opt_flow_conv)

    # conv2
    opt_flow_conv = Conv2D(filters=256, 
                           kernel_size=(5, 5), 
                           strides=(2, 2), 
                           padding="same",
                           name="vggm_block2_conv1")(opt_flow_conv)

    opt_flow_conv = Activation("relu")(opt_flow_conv)
    opt_flow_conv = MaxPooling2D(pool_size=(2, 2), name="vggm_block2_maxpool")(opt_flow_conv)

    # conv3
    opt_flow_conv = Conv2D(filters=512, 
                           kernel_size=(3, 3), 
                           strides=1,
                           activation="relu", 
                           padding="same",
                           name="vggm_block3_conv1")(opt_flow_conv)

    # conv4
    opt_flow_conv = Conv2D(filters=512,
                           kernel_size=(3, 3), 
                           strides=(1, 1), 
                           activation="relu", 
                           padding="same",
                           name="vggm_block4_conv1")(opt_flow_conv)

    # conv5
    opt_flow_conv = Conv2D(filters=512,
                           kernel_size=(3, 3), 
                           strides=(1, 1),
                           activation="relu", 
                           padding="same",
                           name="vggm_block5_conv1")(opt_flow_conv)

    opt_flow_conv = Conv2D(filters=512,
                           kernel_size=(5, 5),
                           strides=(1, 1), 
                           activation="relu", 
                           padding="same",
                           name="vggm_block5_conv2")(opt_flow_conv)

#     opt_flow_conv = MaxPooling2D(pool_size=(2, 2))(opt_flow_conv)
    
    # create a shared model
    tmp_stream = Model(input_tensor, opt_flow_conv, name="vggm")

    return tmp_stream

temporal_stream()


# In[10]:


def spatial_stream(input_shape=(224, 224, 3), verbose=1):
            
    input_tensor = Input(shape=input_shape)        
            
    # Block 1
    x = Conv2D(filters=64, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block1_conv1")(input_tensor)
    
    x = Conv2D(filters=64, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block1_conv2")(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(filters=128, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block2_conv1")(x)
    
    x = Conv2D(filters=128, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block2_conv2")(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(filters=256, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block3_conv1")(x)
    
    x = Conv2D(filters=256, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block3_conv2")(x)
    
    x = Conv2D(filters=256, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block3_conv3")(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block4_conv1")(x)
    
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block4_conv2")(x)
    
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block4_conv3")(x)
    
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block5_conv1")(x)
    
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block5_conv2")(x)
    
    x = Conv2D(filters=512, 
               kernel_size=(3, 3),
               activation="relu",
               padding="same",
               name="block5_conv3")(x)
    
#     x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Create model.
    sp_stream = Model(input_tensor, x, name='vgg16')

    return sp_stream

spatial_stream()


# In[14]:


# function for lambda layer
def antirectifier(list_of_tensors):
    # [?, Time, Width, Height, Channel]
    return K.stack(list_of_tensors, axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 3D tensors
    return shape

# spatiotemporal model
def spatiotemporal_cnn(img_shape=(224, 224), opt_flow_len=10, num_of_classes=10, num_of_snip=5):
    
    img_width = img_shape[0]
    img_height = img_shape[1]
    
    sp_stream = spatial_stream(input_shape=(img_width, img_height, 3))
    tmp_stream = temporal_stream(input_shape=(img_width, img_height, opt_flow_len * 2))

    rgb_imgs = []
    opt_flows = []
    
    input_tensors = []
    for idx in range(num_of_snip):
        rgb_input_name = "rgb_img_{0}".format(idx)
        opt_flow_input_name = "opt_flow_{0}".format(idx)
        rgb_imgs.append(Input(shape=(img_width, img_height, 3), name=rgb_input_name))
        opt_flows.append(Input(shape=(img_width, img_height, opt_flow_len * 2), name=opt_flow_input_name))      
#         input_tensors.append(rgb_imgs[-1])
#         input_tensors.append(opt_flows[-1])
        
    input_tensors = rgb_imgs + opt_flows
    
    print(input_tensors)
    print(len(input_tensors))
        
    output_tensors = []
    for sp_tensor, tmp_tensor in zip(rgb_imgs, opt_flows):
        concat_tensor = concatenate([sp_stream(sp_tensor), tmp_stream(tmp_tensor)])
        output_tensors.append(concat_tensor)

    print(output_tensors)
    
    concat_2_stream = Lambda(function=antirectifier, output_shape=None, mask=None, arguments=None)(output_tensors)
    print("concatenate tensor = ", concat_2_stream.shape)

    n_filters = int(concat_2_stream.shape[4] // 2)

    print(type(n_filters))

    Conv3D_2_stream = Conv3D(filters=n_filters,      # 3d convolution over Time, Height, Width
                             kernel_size=(3, 3, 3),  # specifying the depth (<< notice this), height and width of the 3D convolution window. 
                             strides=(1, 1, 1), 
    #                          padding='valid', 
                             padding='same', 
                             data_format="channels_last", 
    #                          dilation_rate=(1, 1, 1), 
                             activation="relu", 
                             use_bias=True)(concat_2_stream)

    MaxPool3D_2_stream = MaxPooling3D(pool_size=(num_of_snip, 2, 2), strides=(2, 2, 2), padding="valid")(Conv3D_2_stream)
    
    x = Flatten()(MaxPool3D_2_stream)


    x = Dense(2048, name="spatiotemoral_dense1")(x)
    # x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.3)(x)    
    x = Dense(1024, activation="relu", name="spatiotemoral_dense2")(x)
    x = Dense(512, activation="relu", name="spatiotemoral_dense3")(x)
 
    out = Dense(num_of_classes, activation='softmax', name="spatiotemoral_dense4")(x)
    print(out)

    spatiotemporal_cnn = Model(inputs=input_tensors, outputs=out, name="spatiotemporal")
    print(spatiotemporal_cnn.summary())


    return spatiotemporal_cnn

spatiotemporal_cnn()


# In[12]:


def spatiotemporal_cnn_train(num_of_snip=5, 
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
        name_str = "spatiotemporal_cnn"
#         name_str = time_str

    # Callbacks: Save the model.
    saved_model_dir = os.path.join('out', 'checkpoints', name_str)
    
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
            
#     checkpointer = ModelCheckpoint(filepath=os.path.join(saved_model_dir, '{epoch:03d}-{val_loss:.3f}.hdf5'),
    checkpointer = ModelCheckpoint(filepath=os.path.join(saved_model_dir, 'spatiotemproal_cnn_model.hdf5'),
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
    csv_logger = CSVLogger(os.path.join(log_dir, 'spatiotemporal_cnn_training-' + str(timestamp) + '.log'))

    # Callbacks: learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
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

            tmp_model = spatiotemporal_cnn(img_shape=image_shape, opt_flow_len=opt_flow_len, num_of_classes=nb_classes)
            tmp_cnn = multi_gpu_model(tmp_model, gpus=gpus)
        else:
            print("# of GPUS:", gpus)

            tmp_model = spatiotemporal_cnn(img_shape=image_shape, opt_flow_len=opt_flow_len, num_of_classes=nb_classes)
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
                validation_steps=3,
                max_queue_size=20,
                workers=1,
                use_multiprocessing=False)


def main(gpus=1, load_saved_model=False):
    """
    These are the main training settings. 
    Set each before running this file.
    """
    config = {"saved_model": os.path.join('out', 'checkpoints', 'spatiotemporal_cnn', 'spatiotemproal_cnn_model.hdf5')}

    saved_model = None
    class_limit = None  # int, can be 1-101 or None
    num_of_snip = 5 # number of chunks(snippets) used for each video
    opt_flow_len = 10 # number of optical flow frames used
#     image_shape=(224, 224)
    image_shape=(64, 64)
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 64
    nb_epoch = 2222
    name_str = None

    if os.path.exists(config["saved_model"]) and load_saved_model:
        saved_model = config["saved_model"]
        print("using saved model")
    
    spatiotemporal_cnn_train(num_of_snip=num_of_snip, 
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


main(gpus=2)


# In[ ]:


# acc and loss track

import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

log_dir = os.path.join(os.getcwd(), 'out', 'logs', 'spatiotemporal_cnn')
print(log_dir)
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

saved_model = os.path.join(os.getcwd(), 'out', 'checkpoints', 'spatiotemporal_cnn', 'temproal_cnn_model.hdf5')
print(saved_model)

batch_size = 500
class_limit = None  # int, can be 1-101 or None
# yuzhe 20180110
num_of_snip = 1 # number of chunks(snippets) used for each video
opt_flow_len = 10 # number of optical flow frames used
#     image_shape=(224, 224)
image_shape=(64, 64)

# model = temporal_cnn(img_shape, opt_flow_len, n_classes=5)


config={"etc": "./etc", 
        "data_list": "data_list.csv", 
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


# confusion matrix

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

