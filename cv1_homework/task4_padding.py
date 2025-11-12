import numpy as np
import cv2
import matplotlib.pyplot as plt

class Processor:
    def __init__(self):
        pass
    
    def loadpic(self):
        image = 'zrx.jpg'
        cimage = cv2.imread(image)
        
        # 检查图像是否成功加载
        if cimage is None:
            raise FileNotFoundError(f"无法加载图像: {image}")
        
        cimage = cv2.cvtColor(cimage, cv2.COLOR_BGR2RGB)
        return cimage
    
    def padding(self, size):
        image = self.loadpic()
        
        # 正确的彩色图像填充方式
        # 对于形状为 (高度, 宽度, 通道) 的彩色图像
        # 只需要在高度和宽度维度上填充，通道维度不填充
        pad_width = ((size, size), (size, size), (0, 0))
        
        image1 = np.pad(image, pad_width, mode='reflect')
        image2 = np.pad(image, pad_width, mode='edge')
        
        # 显示原始图像和填充后的图像对比
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image1)
        plt.title(f'Reflect Padding\n(size={size})')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image2)
        plt.title(f'Edge Padding\n(size={size})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印形状信息用于调试
        print(f"原始图像形状: {image.shape}")
        print(f"Reflect填充后形状: {image1.shape}")
        print(f"Edge填充后形状: {image2.shape}")

if __name__ == '__main__':
    a = Processor()
    a.padding(200)