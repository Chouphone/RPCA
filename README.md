# 高维数据压缩试验RPCA

#### **简介**:

​	本次实验采取`CNN`进行图像压缩的高位数据降维过程，我们模仿了`auto_encoder`的架构，将`CNN`分为`ComCNN`和`RecCNN`两部分，其中`ComCNN`用以生成一种利用图像进行编码的压缩编码器，保留图像基本的结构信息，使高质量图像可以较准确重建，`RecCNN`用来重新构造解码图像，增强解码图像的质量。

#### 运行环境：

​	建议使用`Python2.7`且需要以下运行环境包，[Tenserflow](http://www.tensorflow.org)，[pickle](https://docs.python.org/3/library/pickle.html) ， [nltk](http://www.nltk.org/install.html)，[scipy](https://www.scipy.org)， [bs4](https://pypi.python.org/pypi/beautifulsoup4/4.3.2)。

#### 组成部分：

- **tradition_jpeg.py**：传统jepg压缩方法封装。主要由压缩，解压缩组成。
- **CNN_model_without_jpeg.py**：高维数据压缩中，不使用传统jpeg方法，只使用CNN网络对图像进行压缩。
- **handle_picture.py**：对图像进行预处理方法封装。
- **CNN_model**：在CNN特征提取之后。使用传统jepg方法再对其进行一次压缩。

