import numpy as np
import cv2
import matplotlib.pyplot as plt


#下面函数用于手动实现二维高斯滤波核
class Image_processor:
    def __init__(self):#别忘了self在这就要写
        pass
    def loadimage(self):
        imagepath="zrx.jpg"
        color_image=cv2.imread(imagepath)
        #以下转换为RGB格式
        color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        return color_image
    def create_gaosi(self,size,sigma):
        kernel=np.zeros((size,size))
        center=size//2#//表示除完后结果为整数5//2=2
        for i in range(size):
            for j in range(size):
                x=i-center
                y=j-center
                gxy=(1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
                kernel[i,j]=gxy
        kernel=kernel/np.sum(kernel)
        """
        别忘了归一化！！！"""
        return kernel
    def compare_kernels(self):
        """这是用来比较高斯核"""
        parameters=[(3,3.0),(5,5.0),(9,7.0)]
        # 以下创建画布显示结果
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        """
        以下注释仅用于我学习（学长学姐请忽略）
        plt.subplots(3, 3, figsize=(12, 10))：

        3, 3: 创建一个3行×3列的子图网格，共9个子图

        figsize=(12, 10): 设置整个图形窗口的尺寸为12英寸宽×10英寸高

         返回值：

        fig: 整个图形窗口对象（Figure对象）

        axes: 一个3×3的numpy数组，包含所有子图对象（Axes对象）
        """
        for i,(size,sigma) in enumerate(parameters):
            mykernel=self.create_gaosi(size,sigma)
            print(f"my kernel is{mykernel}")
            print(f"my_kernel'sum is{mykernel.sum()}")
            cvkernel=cv2.getGaussianKernel(size,sigma)
            cvkernel=np.outer(cvkernel,cvkernel)
            """以下注释也是我用来学习的
            np.outer(a, b)：计算两个向量的外积，结果矩阵的每个元素 result[i,j] = a[i] * b[j]
            """ 
            print(f'opencvgaosi is{cvkernel}')
            print(f"opencvgaosi'sum is{cvkernel.sum()}")
            # 显示手动核
            axes[i, 0].imshow(mykernel, cmap='hot')
            axes[i, 0].set_title(f'my核 {size}x{size}\nσ={sigma}')
            
            # 显示OpenCV核
            axes[i, 1].imshow(cvkernel, cmap='hot')
            axes[i, 1].set_title(f'OpenCV核 {size}x{size}\nσ={sigma}')

        plt.tight_layout()#自动调整子图之间的间距，防止标签重叠
        plt.show()#显示所有创建的图形


if __name__=="__main__":
    pros=Image_processor()
    pros.compare_kernels()