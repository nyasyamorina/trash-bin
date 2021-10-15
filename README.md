# 垃圾代码的最终归处

用来放一些平常用得上 (或者用不上) 的代码.

---

### **LinearHarmonicOscillator.jl**

模拟在 "与距离成正比的力场" 里任意粒子的演化.
在[这里](https://www.bilibili.com/video/BV1HQ4y1B7ia/)可以看到渲染的视频.

参数
 + `φ(x)`: 粒子在 t=0 时的波函数
 + `N`: 对线性谐振子展开级数的数量 *
 + `m ω ħ`: 粒子质量, 力场强度, 普朗克常数 **
 + `x`: 渲染的范围和精度 (左 : 精度 : 右)
 + `time`: 粒子的运动时间 (从0开始到time)
 + `speed`: 视频的播放速度 (视频总时间 = time/speed)
 + `fps outputfile`: No need to explain

*: 取消79行的注释可以看到波函数的能量分布(近似), 当精度足够时, 能量分布末端接近0.

**: 实际上设置这些数值是没有意义的, 经过恰当的空间和时间变换之后, 这些数值都可以变成1.

Note: 渲染视频之后, 视频帧是不会自动删掉的, 所以需要手动把那个文件夹删掉.
视频帧文件夹会以` -> animate frames storage in ...` 给出.

代码最后被注释掉的一段是渲染无损视频的, 但是julia的Cmd会擅自给路径加上单引号,
然而ffmpeg是不能给路径加引号的.

---

### **HydrogenAtom.jl**

展示在氢原子内部的电子轨道.
在[这里](https://www.bilibili.com/video/BV1AL411G7mX/)可以看到渲染的视频.

开头的函数 `HydrogenWaveFunc` 就是输入三个量子数得到电子的波函数. ~~再往下都是垃圾代码了.~~
渲染热图也太慢了, 跑一轮就4个小时, 太难了, 不过也没有优化就是了 ε=ε=ε=┏(゜ロ゜;)┛

需要注意的是, 这个波函数描述的是电子绕着原子核 (坐标原点) 和z轴旋转, 这表明电子轨道, 也就是概率函数 
(波函数模长的平方) 是绕z轴旋转对称的. 至于有时候看到的在x轴或y轴对称的"轨道"实际上是球谐函数的实部,
并不是真正的电子轨道, 老误导人了.

另外, 这簇波函数是归一正交完备的, 也就说任意形式的波函数都可以进行广义傅里叶变换, 
分解得到由这簇波函数线性组合表示的解.

算是一篇含金量很高的代码了 (指数理推导过程). 有机会肯定会出一篇专栏来讲这个东西.
