# yolov5-play
yolov5 played by myself

## 一、前言
- YOLOv5是目标检测领域的重要模型，但**尚未发表正式论文**。其作者Glenn Jocher曾计划在2021年12月1日PyTorch Dev Day前发表论文，否则“吃帽子”，但最终未实现。
- YOLOv5的核心特点是**适合移动端部署**，具有模型小、速度快的优势，在实际应用中广泛使用。


## 二、网络结构概述


<img width="1000" height="440" alt="image" src="https://github.com/user-attachments/assets/3333942b-beaf-4390-a2be-9051291ea0bd" />

YOLOv5包含多个版本，主要差异体现在`depth_multiple`（模型深度）和`width_multiple`（特征图宽度）两个参数，常见版本及特点如下：
| 版本 | 特点 |
|------|------|
| YOLOv5s | 深度最小、特征图宽度最小，是基础版本 |
| YOLOv5m | 在v5s基础上加深、加宽 |
| YOLOv5l | 比v5m更深、更宽 |
| YOLOv5x | 系列中最深、最宽的版本 |

网络整体结构分为4个核心部分：**输入端**、**Backbone（主干网络）**、**Neck（颈部）**、**Head（预测头）**。

<img width="662" height="345" alt="image" src="https://github.com/user-attachments/assets/af1b9c4d-a401-4c34-9d53-91854b71437d" />


## 三、输入端技术细节
输入端主要包含3项关键技术，用于优化数据输入质量和适配模型需求：

### 1. Mosaic数据增强
- **原理**：参考CutMix算法，将**4张图片随机缩放后拼接**成一张图，形成新的训练样本。
- **步骤**：
  1. 随机选取4张图片和拼接基准点坐标（xc，yc）；
  2. 将4张图分别调整尺寸后放置在大图的左上、右上、左下、右下位置；
  3. 同步调整图片标签的坐标映射关系；
  4. 拼接大图并处理超出边界的检测框。
- **优点**：
  - 丰富数据集多样性，增加小目标样本；
  - 增强模型对“非常规语境目标”的检测能力；
  - 提升批归一化层（BN）效果（批样本量越大，均值和方差越接近真实分布）；
  - 专门优化小目标检测性能。


### 2. 自适应锚框计算
- **核心作用**：为不同数据集自动计算最优先验框（anchor），替代YOLOv3/4中单独脚本计算的方式，直接嵌入训练流程。
- **计算过程**：
  1. 提取数据集中所有目标的宽和高；
  2. 将图片等比例缩放到指定大小，转换目标坐标为绝对坐标（乘以缩放后宽高）；
  3. 筛选宽高≥2像素的目标框；
  4. 用k-means聚类得到初始锚框；
  5. 用遗传算法对锚框宽高进行1000次随机变异，保留适应度（通过anchor_fitness计算）更优的结果。
- **灵活性**：可手动关闭该功能，使用自定义锚框。


### 3. 自适应图片缩放
- **目的**：减少图片缩放时的黑边填充，提高推理速度（仅在测试/推理时生效，训练时仍用传统填充）。
- **步骤**：
  1. 计算图片宽、高相对于目标尺寸的缩放系数，选择**较小的系数**（保证图片完全放入目标尺寸）；
  2. 按缩放系数调整图片大小；
  3. 计算黑边填充值，填充值需为32的倍数（因网络经过5次下采样，2⁵=32，避免尺度不匹配）；
  4. 填充灰色（RGB值114,114,114）。


## 四、Backbone（主干网络）
Backbone负责特征提取，核心结构包括Focus和CSP结构。

### 1. Focus结构

<img width="808" height="322" alt="image" src="https://github.com/user-attachments/assets/05b9718f-9ec2-4ebe-b064-a01a65cf5d56" />

- **作用**：在不丢失信息的前提下完成下采样，提升计算效率。
- **操作**：
  - 对输入图片进行切片（每隔1个像素取1个值），将1张图转换为4张互补的子图，通道数从3变为12（如640×640×3→320×320×12）；
  - 经过卷积操作将通道数调整为32（320×320×12→320×320×32）。
- **不足**：对部分设备不友好，开销大；切片对齐要求高，否则易导致模型失效。
- **改进**：新版YOLOv5中，Focus结构被**6×6卷积层**替代（计算量等价，GPU效率更高）。

<img width="386" height="193" alt="image" src="https://github.com/user-attachments/assets/d95da81d-1865-420d-88a0-0b896adefcd6" />


### 2. CSP结构
YOLOv5设计了两种CSP结构，分别用于Backbone和Neck：
- **CSP1_X**：应用于Backbone，由3个卷积层和X个残差组件（Res unit）通过Concat拼接而成，增强特征提取能力；
- **CSP2_X**：应用于Neck，用CBL（Conv+BN+LeakyReLU）替代残差组件，共2×X个CBL，侧重特征融合。

<img width="1295" height="630" alt="image" src="https://github.com/user-attachments/assets/4418ce05-ff8b-4e4a-a746-e2d6b15d8523" />


## 五、Neck（颈部）
- **结构**：采用**FPN+PAN**结构，其中FPN自顶向下传递强语义特征，PAN自底向上传递定位特征。
- **改进**：与YOLOv4相比，Neck中使用CSP2结构替代普通卷积，加强特征融合能力。

<img width="718" height="548" alt="image" src="https://github.com/user-attachments/assets/221cdda3-759e-4b09-808d-d7300b80c2a0" />


## 六、Head（预测头）
负责目标框预测和损失计算，核心技术包括损失函数和非极大值抑制。

### 1. Bounding box损失函数
采用**CIOU_Loss**，在IoU（交并比）基础上，同时考虑目标框的中心点距离、宽高比和重叠度，提升边界框回归精度。


### 2. NMS（非极大值抑制）
- **作用**：消除同一目标上的冗余预测框，保留最佳框。
- **流程**：
  1. 对预测框按置信度降序排序；
  2. 选置信度最高的框为基准，计算与其他框的IoU；
  3. 去除IoU＞阈值的框；
  4. 重复步骤1-3，直到无剩余框。
- **改进（SoftNMS）**：当两个目标距离过近时，不直接删除低置信度框，而是降低其分数，避免漏检（解决NMS对近邻目标的抑制问题）。


## 七、训练策略
- **多尺度训练**：输入尺寸在0.5×目标尺寸到1.5×目标尺寸之间随机选择（需为32的倍数），增强模型对尺度变化的适应性；
- **Warmup预热训练**：初始用小学习率训练（如4个epoch或10000步），再切换到预设学习率，避免模型初期震荡；
- **Cosine学习率下降**：学习率随训练进度按余弦函数逐渐减小，平衡收敛速度和精度；
- **EMA（指数移动平均）**：对模型参数更新施加动量，使权重变化更平滑，提升泛化能力；
- **混合精度训练**：结合FP16和FP32精度，减少显存占用并加速训练（需GPU支持）。


## 八、YOLOv5的核心改进
1. 增加正样本：通过邻域锚框匹配策略，提升目标框匹配效率；
2. 灵活的模型配置：通过depth_multiple和width_multiple参数，可快速调整模型深度和宽度，适应不同部署场景；
3. 内置超参优化：锚框计算、图片缩放等技术嵌入训练流程，减少人工调参成本；
4. 强化小目标检测：通过Mosaic数据增强等策略，提升对小目标的识别能力。


## 九、补充资源
- **源码详解**：涵盖项目目录、推理（detect.py）、训练（train.py）、验证（val.py）、配置文件（yolov5s.yaml）等逐行注释；
- **入门实践**：包括环境配置、数据标注（labelimg）、数据集划分、模型训练及界面开发（pyqt5）；
- **进阶改进**：支持添加注意力机制（SE、CBAM等）、替换主干网络（MobileNetV3、Swin Transformer等）、优化损失函数（EIoU、SIoU等）、改进Neck（BiFPN、AFPN等）等。


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ca88d572-d68c-4e45-8bdc-728dc50fd275" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/724c31d3-7b19-4756-90df-dbfeb0eb63fb" />

![wps2](https://github.com/user-attachments/assets/339dd3c1-a795-4fc6-a25d-deccc366e648)

![wps8](https://github.com/user-attachments/assets/e1865178-6ac0-4cd1-a3b6-a8c4c99633e9)

![wps9](https://github.com/user-attachments/assets/5f669452-2eab-4e82-88fe-d46e487eb6c5)

![wps10](https://github.com/user-attachments/assets/33219a5f-52e9-482b-8976-9425aca2d178)

![wps11](https://github.com/user-attachments/assets/b39091ad-fbe4-49ce-b6d6-0b25cd07db5d)

![wps12](https://github.com/user-attachments/assets/cdb98ef8-da0c-4168-86be-e896daa94378)
