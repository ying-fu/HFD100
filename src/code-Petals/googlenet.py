import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
	layer = nn.Sequential(
		nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
		nn.BatchNorm2d(out_channel, eps=1e-3),
		nn.ReLU(True)
	)
	return layer


class inception(nn.Module):
	def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
		super(inception, self).__init__()
		# 第一条线路
		self.branch1x1 = conv_relu(in_channel, out1_1, 1)

		# 第二条线路
		self.branch3x3 = nn.Sequential(
			conv_relu(in_channel, out2_1, 1),
			conv_relu(out2_1, out2_3, 3, padding=1)
		)

		# 第三条线路
		self.branch5x5 = nn.Sequential(
			conv_relu(in_channel, out3_1, 1),
			conv_relu(out3_1, out3_5, 5, padding=2)
		)

		# 第四条线路
		self.branch_pool = nn.Sequential(
			nn.MaxPool2d(3, stride=1, padding=1),
			conv_relu(in_channel, out4_1, 1)
		)

	def forward(self, x):
		f1 = self.branch1x1(x)
		f2 = self.branch3x3(x)
		f3 = self.branch5x5(x)
		f4 = self.branch_pool(x)
		output = torch.cat((f1, f2, f3, f4), dim=1)
		return output


# 输入为224x224，模型中输入的图片大小多少无所谓，只需要将模型的全连接层处的大小匹配就好


class googlenet(nn.Module):
	def __init__(self, in_channel, num_classes, verbose=False):
		super(googlenet, self).__init__()
		self.verbose = verbose

		self.block1 = nn.Sequential(
			conv_relu(in_channel, out_channel=64, kernel=7, stride=1, padding=0),
			nn.MaxPool2d(3, 1)
		)

		self.block2 = nn.Sequential(
			conv_relu(64, 64, kernel=1),
			conv_relu(64, 192, kernel=3, padding=1),
			nn.MaxPool2d(3, 2)
		)

		self.block3 = nn.Sequential(
			inception(192, 64, 96, 128, 16, 32, 32),
			inception(256, 128, 128, 192, 32, 96, 64),
			nn.MaxPool2d(3, 2)
		)

		self.block4 = nn.Sequential(
			inception(480, 192, 96, 208, 16, 48, 64),
			inception(512, 160, 112, 224, 24, 64, 64),
			inception(512, 128, 128, 256, 24, 64, 64),
			inception(512, 112, 144, 288, 32, 64, 64),
			inception(528, 256, 160, 320, 32, 128, 128),
			nn.MaxPool2d(3, 2)
		)

		self.block5 = nn.Sequential(
			inception(832, 256, 160, 320, 32, 128, 128),
			inception(832, 384, 182, 384, 48, 128, 128),
			nn.AvgPool2d(2)
		)

		self.classifier = nn.Linear(9216, num_classes)

	def forward(self, x):
		x = self.block1(x)
		if self.verbose:
			print('block 1 output: {}'.format(x.shape))
		x = self.block2(x)
		if self.verbose:
			print('block 2 output: {}'.format(x.shape))
		x = self.block3(x)
		if self.verbose:
			print('block 3 output: {}'.format(x.shape))
		x = self.block4(x)
		if self.verbose:
			print('block 4 output: {}'.format(x.shape))
		x = self.block5(x)
		if self.verbose:
			print('block 5 output: {}'.format(x.shape))
		x = x.view(x.shape[0], -1)
		x = self.classifier(x)
		return x