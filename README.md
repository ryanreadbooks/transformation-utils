# Transformation-utils

### 记录一些常用的变换的功能，例如旋转变换等。

### [scipy.spatial.transform提供常用变换的计算API。](http://scipy.github.io/devdocs/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation)

### [pytransform3d库也可以提供相应的计算](https://rock-learning.github.io/pytransform3d/api.html)
---
### 记录如下常用功能
* 由绕xyz三轴旋转角度计算旋转矩阵
* 由轴角的旋转表示形式换成旋转矩阵
* 计算一个向量变成另一个向量所需变换的旋转矩阵
* 计算两个旋转矩阵之间所表示的角度的差值（单位rad）
    * 公式为
    $$\arccos \frac{1}{2}\left( tr\left(R_1 \cdot R_2^T\right)-1\right)
    $$