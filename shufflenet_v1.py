import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import init

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    """
    - 1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def channel_shuffle(x, groups):
    """
    shuffle the feature channel
    :param x: data
    :param groups: divide group numbers
    :return:
    """
    batch_size, channel, height, width = x.data.size()

    channels_per_group = channel//groups
    x = x.view(batch_size, groups, channels_per_group, height, width)

    #shuffle
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):
        '''
        :param groups: the number of group (default 3)
        :param grouped_conv: use the group convolution or not
        :param combine: define the type of ShuffleUnit
        '''
        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        if self.combine == 'add':
            # ShuffleUnit Figure(b)
            self.depthwise = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure(c)
            self.depthwise = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("There is no type of {}".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_conv_group = self.groups if self.grouped_conv else 1

        self.conv1x1_compress = self._make_grouped_conv1x1(self.in_channels,
                                                           self.bottleneck_channels,
                                                           self.first_conv_group,
                                                           batch_norm=True,
                                                           relu=True)
        self.depthwise_con3x3 = conv3x3(self.bottleneck_channels,
                                        self.bottleneck_channels,
                                        stride=self.depthwise,
                                        groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        self.last_con1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels,
                                                             self.out_channels,
                                                             self.groups,
                                                             batch_norm=True,
                                                             relu=False)

    @staticmethod
    def _add(x, output):
        '''
        residual connection
        '''
        return x+output

    @staticmethod
    def _concat(x, output):
        '''
        concatenate the channel axis
        notice: the method is for channel first
        '''
        return torch.cat((x,output), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
                              batch_norm=True, relu=False):
        '''
        Group conv1x1 layers
        '''
        modules = OrderedDict()
        conv = conv1x1(in_channels, out_channels, groups)

        modules['conv1x1'] = conv
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU(True)
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3,
                                    stride=2, padding=1)

        output = self.conv1x1_compress(x)
        output = channel_shuffle(output, self.groups)
        output = self.depthwise_con3x3(output)
        output = self.bn_after_depthwise(output)
        output = self.last_con1x1_expand(output)
        output = self._combine_func(residual,output)

        return output


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        """ShuffleNet constructor.
        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.
        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.num_classes = num_classes

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.in_channels,
                             self.stage_out_channels[1],  # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)
        # Stage 3
        self.stage3 = self._make_stage(3)
        # Stage 4
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
            )
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    """Testing
    """
    test_input = torch.randn(1,3,224,224)
    model = ShuffleNet()
    output = model(test_input)
    print(output.shape)
