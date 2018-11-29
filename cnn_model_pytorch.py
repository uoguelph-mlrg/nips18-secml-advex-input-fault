import torch
import torch.nn as nn
import numpy as np


def truncated_normal(mean=0.0, stddev=1.0, num_samples=1):
    '''
    https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/utils.py
    The generated values follow a normal distribution with specified 
    mean and standard deviation, except that values whose magnitude is 
    more than 2 standard deviations from the mean are dropped and 
    re-picked. Returns a vector of length num_samples
    '''
    samples = []
    for i in range(num_samples):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == num_samples, "something wrong"
    if num_samples == 1:
        return samples[0]
    else:
        return np.array(samples)

    
    
def custom_init_params(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d':
        
        initw = truncated_normal(mean=0, stddev=0.1, num_samples=np.prod(m.weight.shape) )
        initw = initw.reshape(m.weight.shape)
        initw = initw/np.sqrt(1e-7 + np.sum(np.square(initw), axis=(1,2,3)))[:,None,None,None]
        
        if torch.cuda.is_available():
            m.weight.data = torch.nn.Parameter(torch.cuda.FloatTensor(initw))
        else:
            m.weight.data = torch.nn.Parameter(torch.FloatTensor(initw))

        m.bias.data.zero_()
        
    elif classname == 'Linear':
        
        initw = np.random.normal(0, 1.0, size=np.prod(m.weight.shape) )
        initw = initw/np.sqrt(1e-7 + np.sum(np.square(initw)))
        initw = initw.reshape(m.weight.shape)
        if torch.cuda.is_available():
            m.weight.data = torch.nn.Parameter(torch.cuda.FloatTensor(initw))
        else:
            m.weight.data = torch.nn.Parameter(torch.FloatTensor(initw))
        m.bias.data.zero_()
    else:
        pass


class CNN(nn.Module):
    def __init__(self, num_classes, num_filters, num_channels, do_bn):
        super(CNN, self).__init__()

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.do_bn = do_bn

        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=8, stride=2, padding=3)#2
        self.conv2 = nn.Conv2d(num_filters, 2*num_filters, kernel_size=6, stride=2, padding=0)#2
        self.conv3 = nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=5, stride=1, padding=0)#2
        
        self.fc = nn.Linear(256, num_classes) # 256=2*num_filters*(S*S), S=size of conv3 output

        self.act = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(num_filters, eps=1e-05, momentum=0, affine=True, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(2*num_filters, eps=1e-05, momentum=0, affine=True, track_running_stats=False)


    def forward(self, x):
        h_conv1 = self.act(self.conv1(x))

        if self.do_bn:
            h_conv2 = self.act(self.conv2(self.bn1(h_conv1)))
            h_conv3 = self.act(self.conv3(self.bn2(h_conv2)))
        else:
            h_conv2 = self.act(self.conv2(h_conv1))
            h_conv3 = self.act(self.conv3(h_conv2))

        out = self.fc(h_conv3.view(h_conv3.size(0), -1))

        return out