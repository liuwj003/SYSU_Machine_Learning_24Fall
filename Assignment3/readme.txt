手写体数据集包括两个文件：
1.mnist_train.csv
2.mnist_test.csv

mnist_train.csv文件中包含了60000个训练样本，10个类别（0~9）。mnist_test.csv文件包含10000个测试样本，10个类别。文件每一行有785个值，第一列是类别标签（0~9），其余784列是手写字体的像素值。

你需要做的：
不调用现成软件包（如sklearn）情况下，使用numpy，scipy等科学计算包探索 K-Means 和 GMM 这两种聚类算法的性能。