# ”和利时杯“科技创新大赛工作日志

## 4月4日

1. 查阅”铁路信号机视频自动识别与仿真系统“有关的资料和文献，如CN101761038B、CN206579655U两份专利文件，确定了项目背景和大致方向；

2. 设计铁路信号机视频自动识别与仿真系统主要功能页面原型：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/页面原型.jpg" style="zoom:50%;" />

3. 撰写项目申报书并提交申请，**正式报名**；

4. 初步预计后续工作将集中在以下三个模块：

   - 利用树莓派进行环境搭建；
   - 视频识别部分，主要是图像识别方面的学习，尤其是卷积神经网络在图像识别中的应用（这部分是核心）；
   - 设计仿真系统，使用核心的视频识别技术，按照页面原型进行一个完整系统的搭建。



## 4月23日

1. 学习如何从视频中提取图片：[如何从视频中提取图片](https://www.jianshu.com/p/e3c04d4fb5f3)

   主要工具是python的opencv库：

   - 打开摄像头或视频文件：

     ```python
     cap = cv2.VideoCapture(pathIn)	# 打开摄像头或视频文件
     ```

   - 获取视频文件参数：

     ```python
     cap.get()	# 获取视频文件的相关参数，如时间、帧率、大小、当前播放位置（以毫秒为单位）等等
     ```

   - 设置视频文件参数：

     ```python
     cap.set(,)	# 设置视频文件的相关参数
     ```

   - 图片处理函数：

     ```python
     success, image = cap.read()	# 读取视频当前帧，返回读取结果（是否读取成功、读取的图像）
     image = cv2.cvtcolor(image,cv2.COLOR_BGR2GRAY)	# 图像转换为黑白图像
     # ........
     ```

   - 图片输出函数：

     ```
     cv2.imwrite()	# 保存图片，在参数中可设置保存路径、文件名、图片质量、编码格式等
     ```

2. 学习yolov5目标检测模型：[手把手教你使用YOLOV5训练自己的目标检测模型-口罩检测](https://blog.csdn.net/ECHOSON/article/details/121939535)

   - 完成了环境的配置和搭建；
   - 完成了基本测试；




## 5月5日

学习yolov5搭建目标检测平台的相关知识，选择的教程为：[Pytorch 搭建自己的YoloV5目标检测平台](https://www.bilibili.com/video/BV1FZ4y1m777/?spm_id_from=333.788)

在配置环境和进行测试的环节中遇到了如下问题：

**一、环境配置**

由于我的电脑显卡为NVIDIA GTX 1650，因此环境配置参照下面这篇教程：[深度学习环境配置2——windows下的torch=1.2.0环境配置](https://blog.csdn.net/weixin_44791964/article/details/106037141)，但还是有以下问题需要注意：

1. 首先是Anaconda、CUDA、Cudnn的下载和安装，这几步按照教程来即可：

   - CUDA 是 NVIDIA 发明的一种并行计算平台和编程模型，它通过利用图形处理器 (GPU) 的处理能力，可大幅提升计算性能。
   - CUDNN(CUDA Deep Neural Network library)：是NVIDIA打造的针对深度神经网络的加速库，是一个用于深层神经网络的GPU加速库。如果你要用GPU训练模型，cuDNN不是必须的，但是一般会采用这个加速库。

2. 用conda创建一个虚拟环境，在这个虚拟环境中进行操作：

   ```python
   conda create –n py220505 python=3.6		# 创建一个名为py220505的虚拟环境，这里选择的python版本为3.6
   ```

   ```python
   activate py220505		# 激活这个虚拟环境
   ```

3. 安装pytorch库，这里仍然按照教程来就行，注意用的pytorch版本是1.2.0：

   ```python
   # CUDA 10.0
   pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. 安装其他依赖库：

   ```python
   scipy==1.2.1
   numpy==1.17.0
   matplotlib==3.1.2
   opencv_python==4.1.2.30
   torch==1.2.0
   torchvision==0.4.0
   tqdm==4.60.0
   Pillow==8.2.0
   h5py==2.10.0
   ```

   这里建议将上述内容复制到一个requirements.txt文件中，然后直接在创建好的虚拟环境中执行：

   ```python
   pip install -r C:\Users\17262\Desktop\requirements.txt
   ```

5. 关于配置镜像源的问题;

   在上述创建虚拟环境、配置torch、下载安装依赖库步骤中，极有可能遇到无法进行的报错，比如如果在Pycharm中直接安装依赖库，可能会报错：

   ```python
   PackgesNotFoundError: The following packges are not available from current channels:
   ```

   在cmd中安装也会出现类似的报错，或者是下载速度特别慢，这种情况大概率就是镜像源没配置好，因此在安装前首先要进行镜像源的配置：一般清华源、中科大源、阿里源、豆瓣源等等都是比较常见的，我这里用了中科大源有几个库一直装不上，加上清华源才成功配置，配置方法网上一搜就有。

**二、测试环节**

1. 首先，经过上述配置过程。开始按照教程进行测试，可是会报错：

   ```python
   ImportError: TensorBoard logging requires TensorBoard with Python summary writer installed. This should be available in 1.14 or above.
   ```

   此时需要安装tensorboard:

   ```python
   pip install tensorboard==1.14.0
   ```

2. 然后不出意外报错：

   ```python
   ModuleNotFoundError: No module named 'past'
   ```

   这里如果下载past库是不管用的，因为这个past其实是future，解决方法是：

   ```python
   pip install future
   ```

3. 完成之后应该运行就没啥问题了，首先要准备好VOC数据集：

   数据集可以用网上现成的，也可以是自己标注的，但是格式一定要对！！！自己标注数据集可以使用labelimg工具，这个工具直接

   ```python
   pip install labelimg
   ```

   就可以了。

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-05_20-05-11.jpg" style="zoom:40%;" />

   标注过程很简单，不多做赘述，网上也能搜得到。

   这里要求图片的格式为jpg，否则会报错，将png图片批量转化为jpg的方法是：

   在图片所在文件夹内新建一个txt文件，内容为：

   ```python
   ren *.png *.jpg
   ```

   然后将txt格式修改为bat格式，双击运行即可。

4. 然后严格按照教程里的说明来更改相应的路径和参数，正常进行训练和预测就可以了。

   > 注意：教程中提供的github代码中是不包含预训练权重文件的，需要自己根据readme.md文件中的链接去网盘下载。

5. 因为我的GPU内存只有4G，因此在训练的第51轮就会出现“内存爆炸”（CUDA out of memory）现象直接结束，也就是说只能进行冻结阶段的训练，不能完成解冻阶段的训练，解决方法是将train.py文件中解冻环节的Unfreeze_batch_size参数减小为4：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/0b6c3ae52deebf123e06409f8234b68.jpg" style="zoom:50%;" />

   > 在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size;
   >
   > 受到BatchNorm层影响，batch_size最小为2，不能为1;
   >
   > 正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。

6. 这一次我仍然只训练了200张图片，结果是：**可以进行检测，但是效果很差**。

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-05_20-19-07.jpg" style="zoom:50%;" />



## 5月12日

开题答辩，总结如下：

1. 目标检测方面：仍需要对算法原理有一定的了解；
2. 和利时的评委给出的建议是：
   - 增强实际的体验感，考虑与VR联系；
   - 多关注VR这一块，提高沉浸式体验；
   - 需要考虑驾驶体验。
3. 老师的建议是目前先把目标检测做好。



## 5月15日

标注了1061张道路信号图片作为数据集，分类标签只有car和signal两种，用于检测车辆和信号灯，对经过训练后得到的效果进行总结：

1. 设置Freeze_batch_size = 8，Unfreeze_batch_size = 4，训练时长大约为7小时（17：00—24：00）；

2. 由于数据集数量较上次训练增大不少（从200张到1061张），这次的预测效果也要好上不少，道路上的车辆基本都能识别出来，且置信度基本能够达到0.85~0.92，可以说对车辆的识别精度是基本达到要求的：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/ceb21b0ed2e8c5b54b59719168ceeec.jpg" style="zoom:60%;" />

3. 本次训练的结果在预测速度上不是很理想，实时检测的FPS只有7~9，应至少达到25以上才算流畅；

3. 本次训练得到的模型对信号灯的识别效果并不理想，原因大概有两点：

   - **包含信号灯的数据集很少**：数据集中并非每张图片都含有信号灯，事实上1061张图片中可能只有四分之一甚至是五分之一的图片包含信号灯，因此信号灯识别的有效数据集是很少的；

   - **yolov5算法本身存在的缺陷**：yolov5算法在检测小物体方面效果是不太好的，尤其是某一小块区域内存在多个目标，这时就会出现检测对象的丢失，比如下图中，三个信号灯连在一起离得很近，这时基本上只能检测出其中一个或两个信号灯来：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-16_11-03-49.jpg" style="zoom:50%;" />

4. 由于标注不规范以及分类太过笼统的原因，可以看到路边的站台也被识别成了车辆，解决方法是：

   - 加强分类和标注的规范性，增加数据集数量；
   
   - 由于类似站台这样的物体即使被识别成车辆一般置信度也比较低，因此可以通过设置目标置信度confidence参数（该参数位于yolo.py文件），使得只有得分大于置信度的预测框会被保留下来。
   
     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-16_11-11-49.jpg" style="zoom:50%;" />



## 5月21日

一些有关深度学习的背景知识：

1. 计算机视觉的一些基本任务：

   - 图像分类(Image Classifiction)：给定一张图片或一段视频判断里面**包含什么类别的目标**，根据图像的主要内容进行分类；
   - 目标检测(Object Detection)：给定一幅图像，只需要找到一类目标所在的矩形框，它输出的是**目标的边框或标签**；
   - 图像分割(Image Segmentation)：目标分割是检测到图像中的所有目标，分为**语义分割（Semantic-level）**和**实例分割（Instance-level）**，解决“每一个像素属于哪个目标物或场景”的问题，属于像素级的，需要给出属于每一类的所有像素点，而不是矩形框。

   <table>
       <tr>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/1621504521014002309.png" width=500 >图像分类</center></td>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/1621504854626015668.png" width=500 >目标检测</center></td>
       </tr>
   <table>
       <tr>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/1621566263299019213.png" width=500>图像分割（语义分割）</center></td>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/1621567611885057544.png" width=500>图像分割（实例分割）</center></td>
       </tr>

2. 神经网络:

   - 前馈型神经网络(FNN)：直接输入全连接层；
   - 卷积神经网络(CNN)：先进行卷积、池化然后再输入全连接层，算是对FNN的一种改进吧。

3. 目标检测算法：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-22_00-54-19.jpg" style="zoom:65%;" />

   目前的目标检测算法主要包括两阶段(Two-Stage)和一阶段(One-Stage)两类，具体又可分为有先验框和无先验框两种，主流的目标检测算法如上图所示。

   所谓的“**两阶段**”是指：(1)提取候选区和粗分类；(2)目标位置检测和细分类。

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/ce90c567deb785673efffaa189b5423.jpg" style="zoom:40%;" />

   上图是几个两阶段目标检测算法的大致流程图，可以看到从R-CNN到Fast R-CNN到Faster R-CNN，算法都在哪些地方做了改进，更重要的是要理解粗分类和细分类究竟对应着算法的哪些步骤，具体说明可参考文献：*《基于深度学习的带钢表面缺陷检测方法》——第二章“基于深度学习的分类与检测算法”*。

   

## 5月30日

主要进行yolov5网络结构的学习：

yolov5的网络结构包括：Backbone、FPN、YoloHead三个部分，分别对应：**特征提取——特征加强——回归预测**。

<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/0b33c17fb3ac47cfb2b79a80d5b2fbaa.png" style="zoom:50%;" />

1. 主干特征提取网络Backbone：

   Yolov5使用的主干特征提取网络为CSPDarknet，在这个网络中，输入的图片先利用Focus网络对其进行宽度和高度上的压缩以及通道数上的堆叠，然后经过一系列的卷积（Conv)、标准化（BN）、激活函数（SiLU）步骤，利用CSPlayer提取出图片的特征，输出三个不同大小的有效特征层：
   $$
   feat1=(80,80,256)，feat2=(40,40,512)，feat3=(20,20,1024)
   $$

   > 注意：
   >
   > 1. 最新版的yolov5官方已经不再采用Focus网络，而是改用一个6*6的卷积来替代。
   >
   > 2. 这里就解释了**为什么输入网络的图片的尺寸必须要求是32的倍数**：
   >
   >    可以发现，上图中原始图片大小为(640,640,3)，而经过一系列卷积、标准化、激活之后，最后一个有效特征层feature3的大小为(20,20,1024)，也就是说图像的大小变为原来的1/32。
   >
   >    一般的解释是：输入图片在网络中要经过5次下采样，因此要求图片大小必须为2^5=32的整数倍，否则可能图片边缘像素在卷积过程被丢弃掉，也可能导致网络前后层连接出错。

2. 加强特征提取网络FPN：

   得到上述三个有效特征层后，将其输入FPN中进行特征融合和加强提取：

   - feat3=(20,20,1024)的特征层进行1次1X1卷积调整通道后获得P5，P5进行上采样UmSampling2d后与feat2=(40,40,512)特征层进行结合，然后使用CSPLayer进行特征提取获得P5_upsample，此时获得的特征层为(40,40,512)；
   - P5_upsample=(40,40,512)的特征层进行1次1X1卷积调整通道后获得P4，P4进行上采样UmSampling2d后与feat1=(80,80,256)特征层进行结合，然后使用CSPLayer进行特征提取P3_out，此时获得的特征层为(80,80,256)；
   - P3_out=(80,80,256)的特征层进行一次3x3卷积进行下采样，下采样后与P4堆叠，然后使用CSPLayer进行特征提取P4_out，此时获得的特征层为(40,40,512)；
   - P4_out=(40,40,512)的特征层进行一次3x3卷积进行下采样，下采样后与P5堆叠，然后使用CSPLayer进行特征提取P5_out，此时获得的特征层为(20,20,1024)。

   由此，我们得到了三个加强后的特征：
   $$
   P3\_out=(80,80,256),P4\_out=(40,40,512),P5\_out=(20,20,1024)
   $$

3. 分类器与回归器Yolo Head：

   得到前面步骤获取的三个加强特征，就可以对其进行进一步的处理并进行预测了，该部分可以参考：[写给小白的YOLO介绍](https://zhuanlan.zhihu.com/p/94986199?utm_source=pocket_mylist)。

   在这一部分，将要解决以下问题：

   - 如何将加强特征转化为用于预测的数据，以及预测都需要哪些数据，数据格式、大小是怎样的；
   - 边界框Bounding box、真实框Ground truth box、预测框Prediction box、先验框/锚框Anchor Box的概念、区分和确定；
   - 置信度Confidence的求取以及进行得分筛选；
   - 如何进行非极大值抑制；
   - 如何对先验框进行调整得到最终的预测框。





## 5月31日

主要对卷积、标准化、激活函数过程进行学习：

在Yolov5中，Conv2D_BN_SiLU的代码部分如下：

```python
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act  = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
```

**一、卷积**

1. 基本原理：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/202205311531.gif" style="zoom:70%;" />

   卷积的基本原理如上图所示，通过不同大小和分布的卷积核对原始图像进行卷积，可以有效提取出图像的特征，这也是卷积神经网络的核心。

2. 函数解析：

   Yolov5中卷积部分使用了Pytorch中的nn.Conv2d()函数，它的形参如下：

   - in_channels：输入张量的channels；

   - out_channels：期望的输出张量的channels；

   - kernel_size：卷积核的大小；

   - stride：卷积核移动的步长；

   - padding：图像填充的大小；

     > 图像填充的作用是**提高卷积核对边缘部分的数据信息的处理次数，使对边缘信息的提取更加充分**；
     >
     > 以`padding = 1`为例，若原始图像大小为`32x32`，那么padding后的图像大小就变成了`34x34`，而不是`33x33`。

   - group：是否采用分组卷积；

   - bias：是否要添加偏置参数作为可学习参数的一个，默认为True。

**二、批量归一化（BN，Batch_Normalization）**

1. 概念：

   Internal Covariate Shift，内部协变量转移，这是《*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*》一文中提出的概念，指在训练过程中网络参数的变化对网络造成的影响，一方面使得**学习率降低**，另一方面**对每层的参数初始化提出了很高的要求**。

   BN层通常位于卷积层之后、激活层之前，用于对每层的数据进行一个归一化，减少数据的发散程度，降低网络的学习难度，标准化是**处理后的数据服从N(0，1)的标准正态分布**。

   BN对数据的处理方法是：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-05-31_16-22-18.jpg" style="zoom:65%;" />

   总结一下，也就是：
   $$
   y=\frac{x-mean(x)}{\sqrt{Var(x)+eps}}*gamma+beta
   $$

   > 这种处理方法的原理其实就是**中心极限定理**。

2. 函数解析：

   Pytorch中常用nn.BatchNorm2d()函数进行数据批量归一化操作，它的形参如下：

   - num_features：输入数据的通道数，或者说特征数，一般来说BatchNorm2d()函数的输入数据是一个四维数组，大小为batch_size\*num_features\*height\*width；
   - eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5，我们的代码里设置为1e-3；
   - momentum：一个用于运行过程中更新均值和方差的一个估计参数，一般是0.1；
   - affine：当设为true时，会给定可以学习的系数矩阵gamma和beta。

3. 训练过程和测试过程的BN：

   训练过程和测试过程的BN是有一定不同的，区别在于：**训练阶段对每层的数据进行BN后要更新数据的均值和方差，而测试阶段不用**。

   - 训练时：均值、方差分别是**该批次**内数据相应维度的均值与方差；

   - 推理时：均值、方差是**基于所有批次**的期望计算所得。

   当一个模型训练完成之后，它的所有参数都确定了，包括均值和方差，gamma和bata，这时根据全统计量得到的均值和方差被用来进行测试环节的BN。

**三、激活函数**

1. 激活函数是向神经网络中引入**非线性因素**，通过激活函数神经网络就可以拟合各种曲线，其原理大致如下：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_09-33-39.jpg" style="zoom:70%;" />

   激活函数主要分为饱和激活函数（Saturated Neurons）和非饱和函数（One-sided Saturations），非饱和激活函数的优势在于：

   - 非饱和激活函数可以解决梯度消失问题；
   - 非饱和激活函数可以加速收敛。

   常用的饱和激活函数有Sigmoid、Tanh，非饱和激活函数主要有ReLU及其变种。

2. yolov5模型中使用的就是Sigmoid激活函数，该函数的表达式和图像如下：
   $$
   f(x)=\frac{1}{1+e^{-x}}
   $$
   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_09-35-40.jpg" style="zoom:50%;" />

   Sigmoid函数具有以下缺点：

   - **梯度消失**：注意：Sigmoid 函数趋近 0 和 1 的时候变化率会变得平坦，也就是说，Sigmoid 的梯度趋近于 0。神经网络使用 Sigmoid 激活函数进行反向传播时，输出接近 0 或 1 的神经元其梯度趋近于 0。这些神经元叫作饱和神经元。因此，这些神经元的权重不会更新。此外，与此类神经元相连的神经元的权重也更新得很慢。该问题叫作梯度消失。因此，想象一下，如果一个大型神经网络包含 Sigmoid 神经元，而其中很多个都处于饱和状态，那么该网络无法执行反向传播。
   - **不以零为中心**：Sigmoid 输出不以零为中心的,，输出恒大于0，非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢。
   - **计算成本高昂**：exp() 函数与其他非线性激活函数相比，计算成本高昂，计算机运行起来速度较慢。

3. 本次使用的yolov5模型中使用的是Sigmoid函数，具体代码如下：

   ```python
   class SiLU(nn.Module):
       @staticmethod
       def forward(x):
           return x * torch.sigmoid(x)
   ```

   **由于上面提到的Sigmoid激活函数的一些局限性，因此在改进yolov5网络结构的时候可以考虑使用其他非饱和激活函数来替代Sigmoid函数**。

   > 注：激活函数部分参考资料：[深度学习笔记：如何理解激活函数？（附常用激活函数）](https://zhuanlan.zhihu.com/p/364620596)





## 6月3日

进行了第三次模型训练和检测，检测对象为道路信号灯。

1. 本次检测对象分为以下三个类别：

   - red_signal：红灯；
   - green_signal：绿灯；
   - signal：无法判别颜色的所有信号灯。

2. 训练速度：本次训练总时长为10小时12分，数据集从上次的1031张增加到1288张，训练时长由上次的7小时增加到10小时。

3. 检测效果：

   总的来说，在数据集中的检测效果还算理想，但用数据集之外的图片来测试效果就大打折扣了。

   - 在一定条件下置信度能够达到0.9左右，最高甚至能到0.95：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-22-54.jpg" style="zoom:50%;" />

   - 对于近距离的信号灯能够很好的识别，但对于距离较远的信号灯检测效果就不太行了：

     <table>
         <tr>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-27-56.jpg" width=600 >原始图片(远)</center></td>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-27-41.jpg" width=635 >检测效果</center></td>
         </tr>
     <table>
         <tr>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-31-53.jpg" width=600 >原始图片(近)</center></td>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-32-37.jpg" width=635 >检测效果</center></td>
         </tr>

   - 数据集内的图片检测效果好，数据集外的图片检测效果差，模型的泛化能力差：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-03_10-36-16.jpg" style="zoom:40%;" />

     可以看见，在上面这张图片中即使距离很近也无法识别出信号灯。

   - 本次训练模型的mAP达到89.17%，算是较为理想的水准：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-02_22-51-58.jpg" style="zoom:50%;" />
   
     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/mAP.jpg" style="zoom:70%;" />
   
     从mAP可以看出，红灯的检测效果最好，绿灯次之，无法判断颜色的信号灯检测效果最差；而在实际测试中也存在着这样的现象，红灯的识别效果明显比绿灯和其他信号灯更好。
   
4. 分析总结：

   在本次训练检测中，模型的mAP高达89.17%，在数据集中的测试结果较为理想，而在其他环境下几乎无法检测，这是一个比较矛盾的结果，我认为原因主要在于数据集的选择和处理：

   - 调整模型的参数只能使模型在数据集内的检测效果更好，比如能够获得一个较高的mAP，而Yolov5本身就以简单易操作著称，不太需要人为调整网络内部参数（个人观点，有待确认）；

   - 数据集的数量和质量对模型的泛化能力有很大影响，决定了检测效果能否应用到各个场景中，比如在本次训练中，数据集包含的信号灯种类、环境、场景、样式、角度等都是极为单一的，一旦光线较暗信号灯的灯光盖过自然光使得信号灯的轮廓变得模糊不清就难以进行检测了，因此后续有必要增加数据集的数量并提高数据集的质量以保证尽可能涵盖各个场景。

     > 注：模型的泛化能力不仅仅由数据集数量和质量决定，也有其他影响因素。





## 6月13日

进行道路信号灯的实地拍摄，以获得相应的数据集：

1. 线路长度16.88km，包含35个路口，途经商业区、居民区、城郊开发区等多种场景，信号灯包含竖式、横式、立式、悬挂式、圆形、箭头、人形道等多种类别，经过剪辑得到一段时长12分03秒、完全由路口信号灯组成的行车视频：

   <table>
       <tr>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-13_20-32-14.jpg" width=600 >线路1</center></td>
           <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-13_20-40-02.jpg" width=600 >线路2</center></td>
       </tr>

2. 经过视频转图片操作后，得到共21712张图片，分辨率统一为1920*1080，去除极少数不含信号灯的样本后，预计至少可得20000张符合要求的图片:

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-15_09-24-47.jpg" style="zoom:65%;" />

3. 本次拍摄得到数据集仍有一定的局限性，拍摄过程均在白天晴朗条件下完成，未考虑不同天气和不同时间条件因素的影响，如早晨、傍晚、夜间、阴天、雨雪等。





## 6月14日

进行了第四次训练，训练对象主要是道路信号灯：

1. 使用数据集：

   数据集来源为网上下载的数据集，格式为VOC，包含9812张图片，标签种类有18种，具体分类如下所示：

   ```python
   '''
   首字母代表颜色，后面部分代表形状或方向
   '''
   rup
   rright
   rleft
   gup
   gright
   gleft
   gdown
   rbike
   gbike
   rperson
   gperson
   rcircle
   gcircle
   gturn
   rturn
   ycircle
   yright
   yup
   ```

2. 训练时长：

   本次训练起始时间为2022_06_10_14_43_34，结束时间为2022_06_14_18_25_50，总时长99小时42分钟，平均每轮训练用时19.94分钟，训练速度很低，用时远远超过预期；

   同时，在本次训练中监控到GPU的利用率很低，且波动较大，从8%到72%变化不定，GPU的利用率低使得训练时长远远超出正常水平，关于GPU利用率低的问题将在后面详细介绍。

3. 训练效果：

   - mAP：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-14_19-26-50.jpg" style="zoom:70%;" />

     - 本次训练得到模型的mAP为86.91%，低于上次训练，但这并不代表效果很差；

     - 相反，可以看到，前面14类信号灯的AP值几乎全为1，gperson、green_signal、signal这三类的AP值也达到了0.8以上，也就是说由这样一个包含9812张图片的数据集训练得到的模型整体识别效果是很好的；之所以整体的mAP值较低，是因为yup、rbike这两类的AP值为0从而严重拉低了mAP值，这两类AP值为0我猜测可能是因为数据集中关于这两类的图片太少导致的。

   - 实际检测效果：

     本次实验采用之前实际拍摄视频作为实验对象，具体检测结果见：2022_06_14_result.mp4，整体检测效果较好，但也有一些不足：

     - 距离的影响表现得尤为明显：

       明显看到随着距离信号灯越来越近，识别效果越来越好，在较近距离下基本都能维持置信度达到0.9以上：

       <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-15_00-24-11.jpg" style="zoom:40%;" />

       在距离较远的情况下，不仅造成检测对象得分较低，还容易误识别，这些误识别主要集中在形状和方向上，对于信号灯颜色很少出现误识别：

       <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-15_00-31-26.jpg" style="zoom:40%;" />

     - 对于一些易混淆对象（如车灯）容易造成误识别，但这样的情况很少：

       <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-15_00-27-49.jpg" style="zoom:40%;" />
       
     - 对于上次香港某路段的行车视频的识别效果依然不好，其次，对于夜间和光线昏暗的环境下的信号灯的识别效果也非常差。通过交流群中的一些分享和实际识别来看，这部分问题可以统一归为一类：**即信号灯光晕遮挡信号机自身轮廓导致无法识别**。
     
       解决方法：目前能想到的办法是通过人工标注大量夜间或光线较暗环境下的数据集来增强对这一部分信号灯的识别效果。
       
       <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-16_23-07-22.jpg" style="zoom:75%;" />
       
       > 图片来源于交通标志信号灯检测交流群
       
       



## 6月15日

关于第四次**训练速度过慢、时长过长**原因的分析：

第四次训练中，训练时长达到了惊人的99小时42分钟，平均每个epoach耗时约20分钟，这样的训练速度显然是不正常的。而在训练过程中还发现了另一个有趣的现象：**GPU的显存占用很高，利用率却很低，偶尔会猛涨到80%左右，但无法维持；相反，内存占用和CPU的利用率都很高**。

<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-16_23-37-41.jpg" style="zoom:60%;" />

<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-16_23-37-54.jpg" style="zoom:60%;" />

我在网上查找了很多资料，但是对于训练时GPU利用率低的问题的解释却五花八门，结合我们自己的模型，总结原因如下：

1. 数据集太大；
2. 显卡、CPU、内存条等硬件性能限制；
3. 文件I/O、数据保存、日志记录等程序过多，模型处理的大部分时间花在了这上面，而实际使用GPU进行网络参数训练的时间很短；
4. Dataloader用时过长，随着数据集大小增加，数据的加载和预处理时间变得越来越长，造成了GPU一直在等待CPU加载和处理好数据后才正式开始训练。这也解释了为什么CPU的利用率一直接近饱和，以及为什么GPU明明有显存占用但是利用率却很低，而且还会时不时地波动。

总的来说，我更倾向于模型的训练速度慢是上面几种原因的综合作用，同时**CPU和GPU处理数据的速度不匹配是主要影响因素**。

解决方法有如下几种：

1. 增大num_workers，增加同时读取数据的线程数。这种方法我试了一下最多只能设置num_workers=4，否则会”out of memory“；
2. 删除程序中一些无用的日志记录、直方图绘制等步骤，尤其是涉及到Tensorboard的一些操作，网上称这种方法效果显著，能把一轮的时长从十几分钟减少到几十秒，但是这个方法实在太过玄学，因为我们完全不知道代码中哪些部分属于所谓的“无用操作”，找了很久也找不到，修改也就无从谈起。
3. 优化模型的Dataloader部分，这个方法就更玄学了，网上甚至找不到yolov5的数据加载部分如何优化，最多只有Tensorflow架构的相关优化操作，程序里也看不出来是怎么加载的。

总结：目前大致能够确定模型训练速度慢的主要原因，但是还没有比较合适的解决办法。





## 6月24日

进行第五次训练，训练对象依然是道路信号灯：

1. 本次训练在使用的数据集上并没有变化，但是使用的模型由第四次训练的yolov5_s改成了yolov5_m，从而试图达到更高的识别精度；

2. 模型选择：

   - 第四次实验使用的yolov5_s是该系列模型中最小、速度最快、但检测精度也是最低的一个；

   - 本次实验选择yolov5_m也是权衡之后的结果，因为yolov5_l的精度虽然可能更高，但是模型太过庞大，经过实测我的电脑根本无法承受；

   - 哪怕选择的是yolov5_m模型，训练速度依然很慢，平均一轮耗时27分钟，因此我只训练了180轮，在损失函数趋于稳定后即停止训练；

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/e8a34a239f4c1275c66bcf86d0a1747.png" style="zoom:70%;" />

3. 模型的mAP：

   模型的mAP值为87.65%，较第四次训练而言有些许的提升：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-24_14-13-50.jpg" style="zoom:65%;" />

4. 与第四次训练的检测效果对比：

   检测对象仍然使用之前的实拍视频，我利用PR将第四次训练模型的检测结果和第五次训练模型的检测结果进行合成，以便于观察区别，对比视频见*''comparison4&5.mp4''*，分析结果如下：

   - 改用yolov5_m模型后，对于远距离情况下yolov5_s模型识别不到或者识别不精准的对象也能正确地识别了；

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-29_22-40-30.jpg" style="zoom:80%;" />

   - 在距离较近的情况下，yolov5_m模型的置信度却并没有比yolov5_s高太多，大多数时候与之相差无几，有时甚至更低，我的猜想是：

     - 这和本次训练只训练了180轮有关，如果能训练完300轮，或许模型能够更完善；
     - 也有可能在这个距离下yolov5的识别精度已经达到了瓶颈，已经不能通过更改模型来提高。

   - 总体上整个模型的检测效果仍然受限于信号灯距离，从纯视觉检测的角度来说已经不错，但是目前国内外对于红绿灯检测领域很少采用纯视觉检测方法，比如百度Apollo的自动驾驶项目，他们是通过高精度地图先映射出信号灯可能位置，然后在这一块可能位置内进行检测，同时他们使用了长焦镜头和短焦镜头构成了一个类似“集成模型”，具有冗余的效果，同时这也对坐标位置变换提出了很高的要求。

5. 总结：
   - 后续如果想提高识别精度的话，或许只能通过更换模型来训练，然而虽然yolov5对于小目标检测有着一定的缺陷，但是它的快速性和便捷性也是我们难以舍弃的优点；
   - 关于**“光晕遮挡”**的问题，目前还是只能通过增加数据集中此类信号灯的数量来解决，但是我们目前缺少这类夜间信号灯的图片素材，因此在本次训练中并未解决此问题；
   - 如果确定了模型，最终肯定是需要用我们自己的数据集来训练的，尤其是如果做铁路信号机识别的话，基本上不可避免地要自己采集视频并标注，而这一工作是非常耗时的。





## 6月30日（阶段性总结）

一、工作总结

自4月4日提交报名表以来，已经过去将近三个月的时间，在这段时间内，我们对yolov5目标检测算法进行了一定的了解和学习，完成了基于yolov5算法的信号灯检测模型的基本搭建，进行了四次训练检测实验，模型的检测效果已经达到：

- 识别种类多样，基本涵盖目前道路上已知的信号灯种类；
- 识别精度较高，近距离下置信度基本达到0.9~0.95，较远距离下也有较好的识别效果；

但仍存在着一些问题有待解决：

- 检测速度较低，实时视频处理fps只有8~10帧；
- 检测精度仍受限于与信号灯的距离，对于远距离小目标检测效果不佳；
- 信号灯“光晕遮挡”现象使得夜间或光线昏暗条件下几乎无法进行检测。

至此，我们已初步完成了检测部分的工作任务。

二、后续安排

在初步完成检测部分工作后，7月份我们的工作重心将转到信号灯视频检测仿真系统的搭建上来，具体安排是：

1. 基于前一阶段的yolov5检测模型，搭建仿真系统，能够根据视频播放速度换算出行车速度，进而推断出车辆位置，再根据不同的车辆位置确定不同的检测策略（长焦镜头或短焦镜头）；
2. 基于pyqt5设计可视化界面，尽可能做到视频速度切换平滑，检测效果流畅；
3. 在树莓派上完成相关环境的搭建，为项目迁移到树莓派上做好准备；
4. 在进行上述工作的同时，尽可能完善数据集（包括夜间信号灯图片），不断进行训练，提高检测精度，同时想办法提高检测速度。





## 7月7日

1. 关于视频检测仿真系统设计：

   总体目标是将**视频播放速度**和**实际行车速度**关联起来，视频播放速度可以用fps来表达，实际行车速度用v来表示，这二者的关系是：
   $$
   fps=k*v\\
   （即fps与v成正比，k是系数；）
   $$
   而fps有如下限制范围：
   $$
   fps_{min}<=fps<=fps_{max}\\
   （其中，fps_{min}是使得视频保证流畅播放的最低帧率，fps_{max}是指yolov5模型检测的极限帧率）
   $$
   如果采用实时检测的话，可以通过在检测某一帧图像的过程中加入**延时程序**来更改fps，进而表示不同的行车速度：
   $$
   fps=1/(t_d+t_p+t_0)\\
   （其中，
   t_d是yolov5模型检测一张图片的最小时间，t_d=1/fps_{max}；\\
   t_p是对图片的处理时间，如获取该帧、输出检测结果等操作，这些过程几乎不花费时间，即t_p\approx0\\
   t_0是延时程序使得检测过程人为增加的时间，通过更改t_0的大小，可以改变fps大小
   ）
   $$
   举个例子，设k=3，fps_min=25，fps_max=100，则有：

   | 视频播放速度 |        fps        |  实际行车速度  |
   | :----------: | :---------------: | :------------: |
   |    1倍速     |      30帧/s       |     10m/s      |
   |    2倍速     |      60帧/s       |     20m/s      |
   |   0.83倍速   | 25帧/s（fps_min)  | 8.3m/s（v_min) |
   |   3.3倍速    | 100帧/s（fps_max) | 33m/s（v_max） |

   同时，为了实现更好的检测效果，计划在检测过程中加入**长短焦镜头切换**，当距离信号灯较远时使用长焦镜头，当距离信号灯较近时使用短焦镜头，这一操作有两种实现思路：

   - 第一种：假设视频起始时车辆位置为原点，已知在距离原点x0处需要进行长短焦镜头切换，因此在视频播放过程中，要能够根据行车速度和行车时间实时监测车辆的行驶距离x，一旦|x-xt|<=ξ（ξ是一个很小的值，xt表示镜头转换点位置），即将镜头由长焦切换为短焦。这种方法需要同时采集两路输入，由仿真系统判断行车距离x，但这种思路下x的计算会存在一定误差，而且很不方便；
   - 第二种：直接在原视频中切换镜头，即无需仿真系统判断何时进行长短焦镜头切换，这样做既方便又准确，唯一的缺点是此时长短焦切换就完全不受仿真系统控制了，系统也无法得到镜头转换点的具体位置。对于给定的视频输入这样做完全可以，但是如果是实时采集摄像头的话，就很难在合适的位置进行镜头切换了，最多只能加入一个功能由人工进行镜头切换。

   我个人更倾向于第二种思路。

2. 关于视频检测策略：目前想到两种检测策略:

   - 实时检测：

     |          |                                                              |
     | -------- | ------------------------------------------------------------ |
     | 检测方法 | 获取一帧图像，然后对这一帧图像进行检测、处理和输出结果，然后再获取下一帧图像 |
     | 检测特点 | 边检测边输出                                                 |
     | 优点     | 符合项目实际要求，实时检测                                   |
     | 缺点     | 1. 实时检测对检测速度要求很高，以目前的检测速度肯定不可能实现；<br />2. 受限于yolov5模型的检测速度，整个视频播放速度一定存在上下限；<br />3. 对车辆位置的判断存在误差，可能会导致长短焦镜头切换点位置出现偏差。 |

   - 检测完成后一次性输出：

     |          |                                                              |
     | -------- | ------------------------------------------------------------ |
     | 检测方法 | 将原视频检测得到的结果保存，然后一次性输出                   |
     | 检测特点 | 检测与输出分开进行                                           |
     | 优点     | 能很方便地实现播放速度的设置和切换                           |
     | 缺点     | 1. 不符合实际要求；<br />2. 由于检测部分是一次性完成的，所以很难实现检测过程中的长短焦镜头切换。 |

3. 关于检测速度：

   由上面的检测策略可以知道，目前制约整个仿真系统搭建的最大问题是yolov5模型的检测速度，即fps_max。

   现在的fps_max只有10左右，显然不可能满足要求，要提升检测速度，需从硬件软件两方面着手：

   - 硬件上：本来的设想是在树莓派上利用神经计算棒进行加速，然而现在遇到的问题是：
     - 第一，树莓派的GPU既不是Nvidia也不是Intel，由于不知道树莓派的GPU类型，凯璇那边也不知道该配置哪个版本的pytorch；
     - 第二，之前那篇[文章](https://cloud.tencent.com/developer/article/1079212?utm_source=pocket_mylist)哪怕是用了神经计算棒加速后单张图检测时间也达到了300ms，比在自己电脑上还慢；
     - 第三，神经计算棒目前好像不支持pytorch架构，那篇文章里用的是Tensorflow架构；

   - 软件上：目前还没找到到底是什么影响了检测速度，网上的各种yolov5教程的实时检测速度好像都不快。

4. 关于环境搭建：

   凯璇那边目前还是在做迁移，但是第一步的环境配不好，主要是树莓派和我们自己的电脑差距有点大，比如现在不知道GPU类型，也没有显卡驱动，也没有CUDA，所以也不知道该下载对应的哪个pytorch版本。

5. 关于可视化界面设计：

   目前暂定使用pyqt5进行可视化界面设计。





## 7月14日

1. 使用PyQt5进行了简单的界面搭建，目前还比较粗糙，实现了对单张图片的检测：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-07-14_16-08-59.jpg" style="zoom:70%;" />

2. 为了进一步节省检测时间，**将模型加载部分放在了整个GUI的初始化环节** ---> *model_load.py*，在检测图片部分只进行图片的推理检测，避免每检测一次就要重新加载一次模型；

3. 重写了图片检测部分代码 ---> *img_detect.py*，方便直接调用；

4. 受到各方面因素的影响，检测一张图片用时大概在0.25s；其中，由于检测得到的结果是一个np.ndarray数组，不方便直接在窗口中展示，目前的方法是：*先将其保存为图片然后在指定窗口显示，最后销毁此图片*，这一系列文件IO操作耗时大约0.04s，是一个后续需要解决的问题。





## 7月15日

主要研究了如何在PyQt5中播放视频：

1. 首先需要一个QWidget部件，然后将其提升为QVideoWidget；

2. 创建QMediaPlayer对象，指定该对象的播放路径和播放窗口，然后实现最基本的播放，代码如下：

   ```python
   self.player = QMediaPlayer()	# 创建QMediaPlayer对象
   self.player.setVideoOutput(self.widget_2)	# 指定播放窗口为self.widget_2，此窗口一定要是QVideoWidget类
   self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))	# 指定播放路径
   self.player.play()		# 播放
   self.player.pause()		# 暂停
   ```

3. 值得注意的是，如果按上述操作完成后无法正常播放视频，是因为使用的QMediaPlayer，底层是使用DirectShowPlayerService，所以安装一个DirectShow解码器，例如LAV Filters，就可以解决运行出错问题，这里我安装了一个解码器，解决了此问题。

   > 解码器的下载路径：[VideoPlayer_PyQt5](https://github.com/jacke121/VideoPlayer_PyQt5)，这里只要下载里面的LAVFilters-0.74.1-Installer.exe然后双击安装即可。

   效果如下：

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-07-15_16-46-38.jpg" style="zoom:70%;" />



## 7月30日

1. 完成了图片检测和视频检测两大基本功能的实现，最新版代码见https://github.com/wang-10086/yolov5-5.0；

2. 加入了ROI截取功能，能够实现对图片的任意部分进行检测，这样做的好处有两方面

   - **加快了检测速度**，因为截取后的图片尺寸相较于原来小很多：

     以同一张测试图片为例，进行ROI截取后检测时间确实有所降低：

     ```python
     image 1/1 D:\python_project\object_detection\yolov5-u\yolov5-5.0\img_chopped.jpg: 288x640 2 cars, 1 truck, Done. 
     检测用时(0.246s)
     ```

     ```python
     image 1/1 C:\Users\17262\Desktop\test.jpg: 352x640 2 cars, Done. 
     检测用时(0.272s)
     ```

   - **提高了检测精度**，实测进行ROI后检测到的对象变多了：

     <table>
         <tr>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-07-29_23-13-21.jpg" >不进行ROI截取</center></td>
             <td ><center><img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-07-29_23-12-46.jpg" >进行ROI截取</center></td>
         </tr>

3. 目前来看，最大的问题是检测速度太慢，检测一帧图像所用的时间达到0.27s，哪怕是进行ROI截取后也只能达到0.24s左右，是完全不能满足视频检测的流畅性的。



## 8月13日

目前整个系统的最大的两个问题是：

- 检测速度低；
- 无法辨识正确的信号灯。

1. 检测速度：

   关于检测速度方面，我对整个检测过程的各部分用时进行了统计：

   > 设备：CPU Intel i5-9300H，GPU NVIDIA GeForce GTX 1650
   >
   > 视频尺寸：1920*1080

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-08-13_19-30-37.jpg" style="zoom:70%;" />

   - 首先，整个检测过程用时大致集中在四个部分：

     |               检测项目               | 平均用时 |
     | :----------------------------------: | :------: |
     |      img processing(图片预处理)      |  0.120s  |
     |       pre_inference(前置推理)        |  0.068s  |
     | inference+NMS(正式推理+非极大值抑制) |  0.060s  |
     |            total(总时长)             |  0.252s  |

   - 由统计图可知，整个检测过程中耗时最长的是**img processing(图片预处理)**，而真正的检测时间**inference+NMS(正式推理+非极大值抑制)**并不长，各部分时长占比扇形图如下：

     <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-08-13_20-02-23.jpg" style="zoom:75%;" />

   - 同时，在时长统计图中不难发现，检测时长中会出现一些类似于“阶跃信号”的凸起，这是由于在这些时间段，启用了**ROI截取功能**，并因此能得出以下结论：

     启用ROI后，图片尺寸缩小，因此图片预处理的时间大幅增加，前置推理的时间会有小幅度的减少，推理+非极大值抑制的时间会有减少，但是减小幅度可以忽略不计；总的来说，**检测总时长反而会有所增加**。

   - 目前可以考虑的解决方法：

     - 直接**取消pre_inference(前置推理)**，检测速度理论上能够提高 1/3；
     - 改进图片预处理部分的代码，减少文件存取，尤其是ROI截取部分。

2. 尝试加入**多线程处理程序**：

   在视频检测部分加入了一个子线程，但是，使用子线程的优点和缺点都很明显：

   - 优点：使用子线程使得整个程序处理流程更加科学，能够有效避免某一程序耗时过久导致的界面卡死的问题，整体的流畅度也能够大大提高；
   - 缺点：pyqt5的子线程是不能直接控制UI控件的，因此在子线程中既不能**实时**获取ROI、置信度阈值、IOU阈值、期望帧率、检测频率等参数信息，也不能很方便地将检测结果呈现在空间上。
