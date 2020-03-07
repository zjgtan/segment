from torch.nn import *

class VGG16(Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = Sequential(Conv2d(3, 64, 3, padding=100), ReLU(),
                                Conv2d(64, 64, 3, padding=1), ReLU(),
                                MaxPool2d(2, ceil_mode=True))
        self.conv2 = Sequential(Conv2d(64, 128, 3, padding=1), ReLU(),
                                Conv2d(128, 128, 3, padding=1), ReLU(),
                                MaxPool2d(2, ceil_mode=True))

        self.conv3 = Sequential(Conv2d(128, 256, 3, padding=1), ReLU(),
                                Conv2d(256, 256, 3, padding=1), ReLU(),
                                Conv2d(256, 256, 3, padding=1), ReLU(),
                                MaxPool2d(2, ceil_mode=True))

        self.conv4 = Sequential(Conv2d(256, 512, 3, padding=1), ReLU(),
                                Conv2d(512, 512, 3, padding=1), ReLU(),
                                Conv2d(512, 512, 3, padding=1), ReLU(),
                                MaxPool2d(2, ceil_mode=True))

        self.conv5 = Sequential(Conv2d(512, 512, 3, padding=1), ReLU(),
                                Conv2d(512, 512, 3, padding=1), ReLU(),
                                Conv2d(512, 512, 3, padding=1), ReLU(),
                                MaxPool2d(2, ceil_mode=True))
        self.fcn6 = Sequential(Conv2d(512, 4096, 7), Dropout(), ReLU())
        self.fcn7 = Sequential(Conv2d(4096, 4096, 1), Dropout(), ReLU())
        self.score_fn = Conv2d(4096, 21, 1)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X3 = self.conv3(X)
        X4 = self.conv4(X3)
        X5 = self.conv5(X4)
        fcn6 = self.fcn6(X5)
        fcn7 = self.fcn7(fcn6)
        score = self.score_fn(fcn7)
        return X3, X4, score

class FCN8(Module):
    def __init__(self,n_class):
        super(FCN8,self).__init__()
        self.vgg16 = VGG16()
        self.up_sample2=ConvTranspose2d(n_class,n_class,4,2)
        self.up_sample8 = ConvTranspose2d(n_class,n_class,16,8)
        self.score_pool4 = Conv2d(512,n_class,1)
        self.score_pool3 = Conv2d(256,n_class,1)

    def forward(self,X):
        data = X
        pool3,pool4, conv7 = self.vgg16(X)
        A = self.up_sample2(conv7)
        pool4_score = self.score_pool4(pool4)
        pool4_crop = pool4_score[:,:,5:5+A.size()[2],5:5+A.size()[3]].contiguous()
        sumApool4 = pool4_crop + A

        B = self.up_sample2(sumApool4)
        pool3_score = self.score_pool3(pool3)
        pool3_crop = pool3_score[:,:,9:9+B.size()[2],9:9+B.size()[3]].contiguous()
        sumBpool3 = pool3_crop + B

        C = self.up_sample8(sumBpool3)
        C = C[:,:,31:31+data.size()[2],31:31+data.size()[3]].contiguous()
        return C