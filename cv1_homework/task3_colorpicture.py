import numpy as np
import cv2
import matplotlib.pyplot as plt

class GaosiPic():
    def __init__(self):
        pass
    
    def loadimage(self):
        imagepath = "zyy.jpg"
        color_image = cv2.imread(imagepath)
        # 转换为RGB格式
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        return color_image, gray_image
    
    def My_kernel(self, size, sigma):
        mykernel = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                gxy = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
                mykernel[i, j] = gxy
        mykernel = mykernel / np.sum(mykernel)
        return mykernel
    
    def manual_filter_color(self, color_image, size, sigma):
        """
        处理彩色图像的高斯滤波
        分别对R、G、B三个通道进行滤波
        """
        # 分离三个颜色通道
        r_channel = color_image[:, :, 0]
        g_channel = color_image[:, :, 1]
        b_channel = color_image[:, :, 2]
        
        # 分别对每个通道进行高斯滤波
        r_filtered = self.manual_filter(r_channel, size, sigma)
        g_filtered = self.manual_filter(g_channel, size, sigma)
        b_filtered = self.manual_filter(b_channel, size, sigma)
        
        # 合并三个通道
        filtered_color = np.stack([r_filtered, g_filtered, b_filtered], axis=2)
        return filtered_color
    
    def manual_filter(self, image, size, sigma):
        """
        处理单通道图像的高斯滤波（可以是灰度图或单个颜色通道）
        """
        kernel = self.My_kernel(size, sigma)
        kernel_h, kernel_w = kernel.shape
        pad_h = kernel_h // 2  # 填充高度
        pad_w = kernel_w // 2  # 填充宽度
        
        # 对图像进行边界填充
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # 创建结果图像
        result = np.zeros_like(image, dtype=np.float32)
        
        # 对图像进行滤波操作
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_h, j:j+kernel_w]
                result[i, j] = np.sum(region * kernel)
        
        # 确保像素值在0-255之间
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def test_filter(self):
        kernels = [(31, 3.0), (51, 9.0), (71, 22.0)]
        color_image, gray_image = self.loadimage()
        
        # 创建更大的画布来显示彩色和灰度结果
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 显示原始图像
        axes[0, 0].imshow(color_image)
        axes[0, 0].set_title('原始彩色图像')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(gray_image, cmap='gray')
        axes[1, 0].set_title('原始灰度图像')
        axes[1, 0].axis('off')
        
        # 处理并显示滤波后的图像
        for i, (size, sigma) in enumerate(kernels):
            # 处理彩色图像
            color_filtered = self.manual_filter_color(color_image, size, sigma)
            axes[0, i+1].imshow(color_filtered)
            axes[0, i+1].set_title(f'彩色滤波\nsize={size}, σ={sigma}')
            axes[0, i+1].axis('off')
            
            # 处理灰度图像
            gray_filtered = self.manual_filter(gray_image, size, sigma)
            axes[1, i+1].imshow(gray_filtered, cmap='gray')
            axes[1, i+1].set_title(f'灰度滤波\nsize={size}, σ={sigma}')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    a = GaosiPic()
    a.test_filter()