"""
Class for managing our data.
"""
import csv
import numpy as np
import os.path
import random
import threading
from keras.utils import to_categorical
import cv2
#from keras.preprocessing import image

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
#         return next(self.iterator)
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():
    def __init__(self, 
                 num_of_snip=5, # number of snippets 
                 opt_flow_len=10, 
                 image_shape=(224, 224), 
                 original_image_shape=(341, 256), # ?
                 class_limit=None, 
                 config={"etc": "./etc", 
                         "data_list": "data_list.csv", 
                         "img_path": "/home/jovyan/at073-group20/20bn_jester_500/train",                          
#                          "img_path": "/home/jovyan/at073-group20/gesture2img", 
                         "opt_flow_path": "/home/jovyan/at073-group20/20bn_jester_500/optflow/train"}):
#                          "opt_flow_path": "/home/jovyan/at073-group20/gesture2flow_s224"}):

        """
        Constructor.
        opt_flow_len = (int) the number of optical flow frames to consider
        class_limit = (int) number of classes to limit the data to.
        None = no limit.
        """
        # Get the config
        self.config = config
        
        self.num_of_snip = num_of_snip # yuzhe set 1 for test
        self.opt_flow_len = opt_flow_len
        self.image_shape = image_shape
        self.original_image_shape = original_image_shape
        self.class_limit = class_limit
        self.img_path = self.config["img_path"]
        self.opt_flow_path = self.config["opt_flow_path"]
        
        # Get the data.
        self.data_list = self.get_data_list(config)

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning
        self.data_list = self.clean_data_list()
        
    @staticmethod    
    def idx_to_dir_name(dir_idx):
        if dir_idx == 0:
            return "1-1_逆旋-右手"
        elif dir_idx == 1:
            return "1-2_逆旋-左手"
        elif dir_idx == 2:
            return "2-1_正旋-右手"
        elif dir_idx == 3:
            return "2-2_正旋-左手"
        elif dir_idx == 4:
            return "3-1-上下-右手"
        elif dir_idx == 5:
            return "3-2-上下-左手"
        elif dir_idx == 6:
            return "4-1-Thumbup-右手"
        elif dir_idx == 7:
            return "4-2-Thumbup-左手"
        elif dir_idx == 8:
            return "5-1-Stop-右手"
        elif dir_idx == 9:
            return "5-2-Stop-左手"
        
        return None

    @staticmethod
    def get_data_list(config):
        """Load our data list from file."""
        with open(os.path.join(config["etc"], config["data_list"]), 'r') as fin:
            reader = csv.reader(fin)
            data_list = list(reader)

        return data_list

    def clean_data_list(self):
        data_list_clean = []
        for item in self.data_list:
            if item[1] in self.classes:
                data_list_clean.append(item)

        return data_list_clean

    def get_classes(self):
        """
        Extract the classes from our data, '\n'. 
        If we want to limit them,
        only return the classes we need.
        """
        classes = []
        for item in self.data_list:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""

        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert label_hot.shape[0] == len(self.classes)

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data_list:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    @threadsafe_generator
    def stack_generator(self, batch_size, train_test, name_str="N/D"):
        """
        Call this method after creating an DataSet object
        Return a generator of optical frame stacks that we can use to train on. There are
        a couple different things we can return:
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test() # type of train, test is list
        data_list = train if train_test == 'train' else test

        idx = 0
        
        print("\nDataSet-144: Creating %s generator with %d samples.\n" % (train_test, len(data_list)))

        while 1:
            idx += 1
#             print("Generator yielding batch No.%d" % idx)
#             if(train_test == 'test'):
#                 print("Validating for job: %s" % name_str)
            stacked_img = []
    
            stacked_list = [[] for _ in range(2 * self.num_of_snip)]
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                stack = []
                img_stack = []

                # Get a random sample.
                # columns of row: train or test, class, video dir name, random number
                row = random.choice(data_list)
                
                # Get the stacked optical flows from disk.
#                 gen_data = self.get_stacked_opt_flows(row, train_test)
                stack = self.get_stacked_opt_flows(row, train_test)

                if stack is None:
                    print("warning: return None in {0}.".format(row))
                    continue
                else:
                    for tmp_idx, tmp_list in enumerate(stacked_list):
                        tmp_list.append(stack[tmp_idx])
#                     img_stack = gen_data[0]
#                     stack = gen_data[1]

#                 stacked_img.append(img_stack)
#                 X.append(stack)
                y.append(self.get_class_one_hot(row[1])) # which class
            
#             stacked_img = np.concatenate(stacked_img, axis=0)

            
#             X = np.concatenate(X, axis=0)

#             X = np.array(X)
            X = [np.stack(tmp_list) for tmp_list in stacked_list]
#             print("check-data-171, shape of X", [tmp_.shape for tmp_ in X])

            y = np.array(y)
            y = np.squeeze(y)

#             yield [stacked_img, X], y 
            yield X, y 

    
    def get_stacked_opt_flows(self, row, train_test, crop='corner', val_aug='center'):
        # crop options for training: corner, random
        # augmentation options for testing: resize, center

        img_stack = [] # stacked image
#         sub_dir = self.idx_to_dir_name(int(row[3]))
        img_dir = os.path.join(self.img_path, row[1], row[2])

        opt_flow_stack = [] # stacked optical flow
        opt_flow_dir_x = os.path.join(self.opt_flow_path, 'u', row[1], row[2])
        opt_flow_dir_y = os.path.join(self.opt_flow_path, 'v', row[1], row[2])

#         print("temporal-data-180, train or test: {0}\n".format(train_test))
#         print("temporal-data-182, opt_flow_dir_x: {0}\n".format(opt_flow_dir_x))
#         print("temporal-data-184, opt_flow_dir_y: {0}\n".format(opt_flow_dir_y))

        # spatial parameters
#         if train_test == 'train':
#             if crop == 'random':
#                 # crop at center and four corners randomly for training
#                 left, top = random.choice([[0, 0], 
#                                            [0, self.original_image_shape[1] - self.image_shape[1]], 
#                                            [self.original_image_shape[0] - self.image_shape[0], 0], 
#                                            [self.original_image_shape[0] - self.image_shape[0], self.original_image_shape[1] - self.image_shape[1]],
#                                            [int((self.original_image_shape[0] - self.image_shape[0]) * 0.5), int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)]])
#             else:
#                 # random crop for training set
#                 left = int((self.original_image_shape[0] - self.image_shape[0]) * random.random())
#                 top = int((self.original_image_shape[1] - self.image_shape[1]) * random.random())
#         else:
#             # crop at center for validation
#             left = int((self.original_image_shape[0] - self.image_shape[0]) * 0.5)
#             top = int((self.original_image_shape[1] - self.image_shape[1]) * 0.5)
            
#         right = left + self.image_shape[0] 
#         bottom = top + self.image_shape[1]

        # [top, bottom] and [left, right] have been setted up already
        # temporal parameters
#         assert len(os.listdir(opt_flow_dir_x)) == len(os.listdir(opt_flow_dir_y))
        
        #if not len(os.listdir(opt_flow_dir_x)) == len(os.listdir(opt_flow_dir_y)):
        #    print("AssertionError-268: # of files does not equal in ", opt_flow_dir_x, opt_flow_dir_y)

        total_frames = len(os.listdir(opt_flow_dir_x))     
        
#         win_len = (total_frames - self.opt_flow_len) // self.num_of_snip # starting frame selection window length
#         win_len = self.opt_flow_len 

#         if train_test == 'train':
#             start_frame = int(random.random() * win_len) + 1
#         else:
#             start_frame = int(0.5 * win_len) + 1

       
    
        start_frame = (self.opt_flow_len // 2 + 1) + 1
        end_frame = total_frames - (self.opt_flow_len // 2 + 1)
        
#         if total_frames < self.opt_flow_len * self.num_of_snip:
        if end_frame < start_frame:

            print("warning-data-265: (# of opt flow =) {0} < {1} in dir {1}".format(total_frames, self.opt_flow_len * self.num_of_snip, opt_flow_dir_x))
            return None
        
#         print(block_sep_line)
#         print("temporal-data-216, num_of_snip: {0}\n{1} ".format(self.num_of_snip, sep_line))
#         print("temporal-data-218, opt_flow_len: {0}\n{1} ".format(self.opt_flow_len, sep_line))
#         print("temporal-data-220, total_frames: {0}\n{1} ".format(total_frames, sep_line))
#         print(block_sep_line)

        # selected images and optical flow frames
    
        random_imgs = np.sort(np.random.choice(a=np.arange(start_frame, end_frame + 1), size=self.num_of_snip, replace=True))
        
        rgb_frames = []
        frames = [] 
        
        for i in range(self.num_of_snip):
            # yuzhe 20180102
#             start_idx = start_frame + self.opt_flow_len * i
#             end_idx = start_frame + self.opt_flow_len * (i + 1)
            
            start_idx = random_imgs[i]
            rgb_frames.append(start_idx )
            count_opt_flow = 0
            tmp_opt_flow = []
            for tmp_idx in range(start_idx - self.opt_flow_len // 2, total_frames):
                count_opt_flow += 1
                 

                tmp_opt_flow.append(tmp_idx)
                if count_opt_flow == self.opt_flow_len:
                    break
#             if end_idx > total_frames:
#                 continue
# #                 return None
            
#             rgb_frames.append((start_idx + end_idx) // 2)
#             frames += range(start_idx, end_idx)
#             frames.append([tmp_idx for tmp_idx in range(start_idx, end_idx)])
            frames.append(tmp_opt_flow)

        # flip
#         if train_test == 'train' and random.random() > 0.5:
#             flip = True
#         else:
#             flip = False

        # loop over images 
        for i_img in rgb_frames:
            img = None
            img = cv2.imread(img_dir + "/{0:05d}.jpg".format(i_img))
            img = cv2.resize(img, self.image_shape)

            img = np.array(img)
            img = img - np.mean(img)
            
            img = img / 255. # [H, W, 3]
            img_stack.append(img)
        
#         img_stack = np.array(img_stack)
            
        # loop over frames
        for opt_flow_frames in frames:
            opt_flow = []
            
            for i_frame in opt_flow_frames:
                try:
                    # horizontal components
                    img = None # reset to be safe
    #                 img = cv2.imread(opt_flow_dir_x + '/frame' + "%06d"%(i_frame) + '.jpg', 0)
#                     img = cv2.imread(opt_flow_dir_x + '/u_optlflow' + "%06d"%(i_frame) + '.jpg', 0)
                    img = cv2.imread(opt_flow_dir_x + "/optlflow{0:05d}.jpg".format(i_frame), 0)

                    # mean substraction 
                    img = np.array(img)
                    img = img - np.mean(img)

#                     if train_test == 'train' or val_aug == 'center':
#                         # crop
#                         img = img[left:right, top:bottom]
#                         try:
#                             img = cv2.resize(img, self.image_shape)
#                         except:
#                             print(img.shape)
#                             pass
#                     else:
#                         #resize
                    img = cv2.resize(img, self.image_shape)

                    img = img / 255. # normalize pixels 
#                     if flip:
#                         img = -img

    #                 opt_flow_stack.append(img)
                    opt_flow.append(img)

                except:
                    print("DataSet-220, img_x: {0}".format(opt_flow_dir_x + "/optlflow{0:06d}.jpg".format(i_frame)))            
                    print("DataSet-222, shape of img_x: {0}".format(img.shape))

                try:
                    # vertical components
                    img2 = None # reset to be safe
    #                 img2 = cv2.imread(opt_flow_dir_y + '/frame' + "%06d"%(i_frame) + '.jpg', 0)
#                     img2 = cv2.imread(opt_flow_dir_y + '/v_optlflow' + "%06d"%(i_frame) + '.jpg', 0)                    
                    img2 = cv2.imread(opt_flow_dir_y + "/optlflow{0:05d}.jpg".format(i_frame), 0)

        
                    # mean substraction 
                    img2 = np.array(img2)
                    # yuzhe 20180110
    #                 img2 = np.swapaxes(img2, 0, 1)
                    img2 = img2 - np.mean(img2)

#                     if train_test == 'train' or val_aug == 'center':
#                         # crop
#                         img2 = img2[left:right, top:bottom]
#                         try:
#                             img2 = cv2.resize(img2, self.image_shape)
#                         except:
#                             print(img2.shape)
#                             pass
#                     else:
#                         #resize
                    img2 = cv2.resize(img2, self.image_shape)

                    img2 = img2 / 255. # normalize pixels 
                    opt_flow.append(img2)
    #                 opt_flow_stack.append(img2)

                except:
                    print("DataSet-238 img_y: {0}".format(opt_flow_dir_y + "/optlflow{0:06d}.jpg".format(i_frame)))
                    print("DataSet-240 shape of img_y: {0}".format(img2.shape))

            opt_flow = np.array(opt_flow)
            opt_flow = np.swapaxes(opt_flow, 0, 1)
            opt_flow = np.swapaxes(opt_flow, 1, 2) # [H, W, self.opt_flow_len]

            opt_flow_stack.append(opt_flow)   

    #         opt_flow_stack = np.array(opt_flow_stack)
#             print("DataSet-386 shape of opt_flow_stack: {0}".format(opt_flow_stack.shape))
#             print("DataSet-386 shape of opt_flow_stack: {0}".format(len(opt_flow_stack))

    #         opt_flow_stack = np.swapaxes(opt_flow_stack, 0, 1)
    #         opt_flow_stack = np.swapaxes(opt_flow_stack, 1, 2)
    #         print("DataSet-312 shape of opt_flow_stack: {0}".format(opt_flow_stack.shape))


        # random horizontal flip for training sets
#         if flip:
#             opt_flow_stack = np.flip(opt_flow_stack, 0)

#         return [img_stack, opt_flow_stack]

        # return a list that length is 2 * self.num_of_snip
        return img_stack + opt_flow_stack