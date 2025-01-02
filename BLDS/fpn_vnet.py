import torch
import torch.nn as nn


class ResidualBlock3(nn.Module):
    """ residual block with variable number of convolutions """

    def __init__(self, channels, ksize, padding, num_convs):
        super(ResidualBlock3, self).__init__()

        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=True))
            else:
                layers.append(ConvBnRelu3(channels, channels, ksize, padding, do_act=False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        output = self.ops(input)
        return self.act(input + output)


class ConvBnRelu3(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(ConvBnRelu3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.bn(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class UpBlock(nn.Module):
    """ parametric up-sampling block """

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class DownBlock(nn.Module):
    """ parametric down-sampling block """

    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.bn(self.conv(input)))
        return out


class FPNVnet(nn.Module):
    """ Feature Pyramid Network using VNet as backbone """

    def __init__(self, in_channels, num_anchors, down_ratios=[2,4,8,16], loss_func=None, output_loss=False):
        """ constructor
        :param num_anchors      the number of anchors at each location and at each scale
        :param loss_func        the loss function
        :param output_loss      whether to output loss value
        """
        super(FPNVnet, self).__init__()

        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.down_ratios = down_ratios

        # VNET - down-sampling pathway

        # stage 1 - (sx,   sy,   sz)
        self.left_block1 = ConvBnRelu3(in_channels=self.in_channels, out_channels=16, ksize=3, padding=1)

        # stage 2 - (sx/2, sy/2, sz/2)
        self.down_block2 = DownBlock(in_channels=16, out_channels=32)
        self.left_block2 = ResidualBlock3(channels=32, ksize=3, padding=1, num_convs=1)

        # stage 3 - (sx/4, sy/4, sz/4)
        self.down_block3 = DownBlock(in_channels=32, out_channels=64)
        self.left_block3 = ResidualBlock3(channels=64, ksize=3, padding=1, num_convs=2)

        # stage 4 - (sx/8, sy/8, sz/8)
        self.down_block4 = DownBlock(in_channels=64, out_channels=128)
        self.left_block4 = ResidualBlock3(channels=128, ksize=3, padding=1, num_convs=3)

        # stage 5 - (sx/16, sy/16, sz/16)
        self.down_block5 = DownBlock(in_channels=128, out_channels=256)
        self.left_block5 = ResidualBlock3(channels=256, ksize=3, padding=1, num_convs=3)

        # VNET - up-sampling pathway

        # stage 4 - (sx/8, sy/8, sz/8)
        self.up_block4 = UpBlock(in_channels=256, out_channels=128)
        self.right_block4 = ResidualBlock3(channels=256, ksize=3, padding=1, num_convs=3)

        # stage 3 - (sx/4, sy/4, sz/4)
        self.up_block3 = UpBlock(in_channels=256, out_channels=64)
        self.right_block3 = ResidualBlock3(channels=128, ksize=3, padding=1, num_convs=3)

        # stage 2 - (sx/2, sy/2, sz/2)
        self.up_block2 = UpBlock(in_channels=128, out_channels=32)
        self.right_block2 = ResidualBlock3(channels=64, ksize=3, padding=1, num_convs=2)

        # box proposal heads
        self.head5 = nn.Sequential(
            ConvBnRelu3(in_channels=256, out_channels=256, ksize=3, padding=1),
            nn.Conv3d(in_channels=256, out_channels=num_anchors * 7, kernel_size=1)
        )

        self.head4 = nn.Sequential(
            ConvBnRelu3(in_channels=256, out_channels=256, ksize=3, padding=1),
            nn.Conv3d(in_channels=256, out_channels=num_anchors * 7, kernel_size=1)
        )

        self.head3 = nn.Sequential(
            ConvBnRelu3(in_channels=128, out_channels=128, ksize=3, padding=1),
            nn.Conv3d(in_channels=128, out_channels=num_anchors * 7, kernel_size=1)
        )

        self.head2 = nn.Sequential(
            ConvBnRelu3(in_channels=64, out_channels=64, ksize=3, padding=1),
            nn.Conv3d(in_channels=64, out_channels=num_anchors * 7, kernel_size=1)
        )

        self.loss_func = loss_func
        self.output_loss = output_loss

    def reshape_target(self, bp):
        """ reshape the network output to target-like shape """

        batchsize = bp.size(0)
        channels = bp.size(1)
        assert channels == self.num_anchors * 7
        depth, height, width = bp.size(2), bp.size(3), bp.size(4)
        bp = bp.view(batchsize, channels, -1)
        bp = bp.transpose(1, 2).contiguous().view(batchsize, depth, height, width, self.num_anchors, 7)
        return bp

    def forward(self, input, targets=None):
        """ forward pass to compute box proposals """

        left1 = self.left_block1(input)
        left2 = self.left_block2(self.down_block2(left1))   # down ratio 2
        left3 = self.left_block3(self.down_block3(left2))   # down ratio 4
        left4 = self.left_block4(self.down_block4(left3))   # down ratio 8
        left5 = self.left_block5(self.down_block5(left4))   # down ratio 16

        # down ratio 8
        right4 = torch.cat((left4, self.up_block4(left5)), dim=1)
        right4 = self.right_block4(right4)

        # down ratio 4
        right3 = torch.cat((left3, self.up_block3(right4)), dim=1)
        right3 = self.right_block3(right3)

        # down ratio 2
        right2 = torch.cat((left2, self.up_block2(right3)), dim=1)
        right2 = self.right_block2(right2)

        bp5 = self.head5(left5)
        bp5 = self.reshape_target(bp5)

        bp4 = self.head4(right4)
        bp4 = self.reshape_target(bp4)

        bp3 = self.head3(right3)
        bp3 = self.reshape_target(bp3)

        bp2 = self.head2(right2)
        bp2 = self.reshape_target(bp2)

        pred_targets = []
        pred_targets_head = [bp2, bp3, bp4, bp5]
        for idx, ratio in enumerate([2, 4, 8, 16]):
            if ratio in list(self.down_ratios):
                pred_targets.append(pred_targets_head[idx])

        if self.output_loss:
            assert [tuple(pred.shape) for pred in pred_targets] == [tuple(gt.shape) for gt in targets], \
                'shape of predict tensor and groundtruth tensor must match'
            return self.loss_func(pred_targets, targets)
        else:
            return pred_targets

    @staticmethod
    def max_stride():
        return 16

