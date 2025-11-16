'''
以下代码完全由AI完成，请学长学姐忽略此文件
'''


# 导入需要的工具包
import numpy as np  # 用于数学计算和数组操作
import cv2  # 用于图像处理
import matplotlib.pyplot as plt  # 用于画图显示结果

"""
这个程序要完成三个任务：
1. 手动创建高斯滤波核，并与OpenCV的进行比较
2. 手动实现图像滤波（模糊）效果
3. 手动实现两种边界填充方式
"""

class ImageProcessor:
    def __init__(self):
        # 初始化类，暂时不需要做什么
        pass
    
    # ==================== 任务1：创建高斯滤波核 ====================
    
    def create_gaussian_kernel(self, size, sigma):
        """
        手动创建二维高斯滤波核
        高斯核就像一个"权重模板"，中心权重最大，四周权重逐渐变小
        
        参数说明：
        size: 滤波核的大小，比如5就是5x5的网格
        sigma: 控制模糊程度的参数，越大越模糊
        """
        # 创建一个空的网格（全零）
        kernel = np.zeros((size, size))
        center = size // 2  # 找到中心点位置
        
        # 遍历网格中的每个点
        for i in range(size):
            for j in range(size):
                # 计算当前点距离中心点的位置
                x_distance = i - center
                y_distance = j - center
                
                # 使用高斯函数公式计算权重
                # 这个公式让中心点权重最大，离中心越远权重越小
                gaussian_value = (1 / (2 * np.pi * sigma**2)) * np.exp(
                    -(x_distance**2 + y_distance**2) / (2 * sigma**2))
                
                kernel[i, j] = gaussian_value
        
        # 让所有权重加起来等于1（归一化）
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def compare_kernels(self):
        """比较手动创建的高斯核和OpenCV生成的高斯核"""
        print("开始比较高斯滤波核...")
        
        # 定义三组不同的参数（大小和sigma）
        parameters = [
            (3, 0.5),   # 小核，轻微模糊
            (5, 1.0),   # 中核，中等模糊  
            (7, 1.5)    # 大核，重度模糊
        ]
        
        # 创建画布显示结果
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        
        for i, (size, sigma) in enumerate(parameters):
            print(f"\n第{i+1}组参数 - 大小: {size}x{size}, sigma: {sigma}")
            
            # 1. 手动创建高斯核
            my_kernel = self.create_gaussian_kernel(size, sigma)
            print(f"手动核总和: {my_kernel.sum():.6f} (应该接近1)")
            
            # 2. 用OpenCV创建高斯核
            opencv_kernel = cv2.getGaussianKernel(size, sigma)
            opencv_kernel = np.outer(opencv_kernel, opencv_kernel)  # 变成二维
            print(f"OpenCV核总和: {opencv_kernel.sum():.6f}")
            
            # 3. 计算差异
            difference = np.abs(my_kernel - opencv_kernel)
            print(f"最大差异: {difference.max():.8f}")
            
            # 显示手动核
            axes[i, 0].imshow(my_kernel, cmap='hot')
            axes[i, 0].set_title(f'手动核 {size}x{size}\nσ={sigma}')
            
            # 显示OpenCV核
            axes[i, 1].imshow(opencv_kernel, cmap='hot')
            axes[i, 1].set_title(f'OpenCV核 {size}x{size}\nσ={sigma}')
            
            # 显示差异
            axes[i, 2].imshow(difference, cmap='hot')
            axes[i, 2].set_title(f'差异图\n最大差异: {difference.max():.6f}')
        
        plt.tight_layout()
        plt.show()
    
    # ==================== 任务2：手动实现滤波操作 ====================
    
    def manual_filter(self, image, kernel):
        """
        手动实现图像滤波
        就像用一个带权重的"印章"在图像上滑动，每个像素都受到周围像素的影响
        
        参数说明：
        image: 要处理的图像
        kernel: 滤波核（权重模板）
        """
        # 获取滤波核的大小
        kernel_h, kernel_w = kernel.shape
        pad_h = kernel_h // 2  # 上下需要填充的宽度
        pad_w = kernel_w // 2  # 左右需要填充的宽度
        
        # 对图像进行边界填充（镜像填充）
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # 创建结果图像（全零）
        result = np.zeros_like(image, dtype=np.float32)
        
        # 对每个像素进行滤波操作
        for i in range(image.shape[0]):      # 遍历每一行
            for j in range(image.shape[1]):  # 遍历每一列
                # 取出当前像素周围的区域
                region = padded_image[i:i+kernel_h, j:j+kernel_w]
                # 计算加权平均（区域 × 权重核）
                result[i, j] = np.sum(region * kernel)
        
        # 确保像素值在0-255之间
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def test_filtering(self):
        """测试不同的滤波效果"""
        print("\n开始测试滤波效果...")
        
        # 创建一个简单的测试图像（黑白格子）
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[20:40, 20:40] = 255  # 白色方块
        test_image[60:80, 60:80] = 255  # 白色方块
        
        # 定义不同的滤波参数
        kernels = [
            (3, 0.5, "小核轻微模糊"),
            (5, 1.0, "中核中等模糊"), 
            (7, 1.5, "大核重度模糊")
        ]
        
        # 创建画布显示结果
        fig, axes = plt.subplots(2, len(kernels)+1, figsize=(15, 8))
        
        # 显示原图
        axes[0, 0].imshow(test_image, cmap='gray')
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        axes[1, 0].axis('off')  # 下面留空
        
        for i, (size, sigma, title) in enumerate(kernels):
            # 创建高斯核
            kernel = self.create_gaussian_kernel(size, sigma)
            
            # 进行滤波
            filtered_image = self.manual_filter(test_image, kernel)
            
            # 显示滤波结果
            axes[0, i+1].imshow(filtered_image, cmap='gray')
            axes[0, i+1].set_title(title)
            axes[0, i+1].axis('off')
            
            # 显示使用的滤波核（放大显示）
            kernel_display = cv2.resize(kernel, (80, 80), interpolation=cv2.INTER_NEAREST)
            axes[1, i+1].imshow(kernel_display, cmap='hot')
            axes[1, i+1].set_title(f'滤波核 {size}x{size}')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # ==================== 任务3：手动实现边界填充 ====================
    
    def manual_padding(self, image, pad_size, mode='zero'):
        """
        手动实现边界填充
        因为滤波时边缘像素周围没有足够的像素，所以需要在边界外填充一些像素
        
        参数说明：
        image: 要填充的图像
        pad_size: 填充的宽度
        mode: 填充方式 'zero'零填充, 'reflect'镜像填充
        """
        if mode == 'zero':
            # 零填充：在边界外填充0（黑色）
            padded = np.pad(image, pad_size, mode='constant', constant_values=0)
            print("使用零填充方式")
            
        elif mode == 'reflect':
            # 镜像填充：像镜子一样反射边界内的像素
            padded = np.pad(image, pad_size, mode='reflect')
            print("使用镜像填充方式")
            
        elif mode == 'edge':
            # 边缘填充：重复边界像素
            padded = np.pad(image, pad_size, mode='edge')
            print("使用边缘填充方式")
            
        else:
            # 默认使用零填充
            padded = np.pad(image, pad_size, mode='constant', constant_values=0)
        
        return padded
    
    def test_padding(self):
        """测试不同的填充方式"""
        print("\n开始测试边界填充...")
        
        # 创建一个小图像便于观察效果
        small_image = np.array([
            [100, 100, 100, 100],
            [100, 200, 200, 100], 
            [100, 200, 200, 100],
            [100, 100, 100, 100]
        ], dtype=np.uint8)
        
        # 测试三种填充方式
        padding_methods = ['zero', 'reflect', 'edge']
        titles = ['零填充', '镜像填充', '边缘填充']
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        
        # 显示原图
        axes[0, 0].imshow(small_image, cmap='gray')
        axes[0, 0].set_title('原始图像 (4x4)')
        axes[0, 0].axis('off')
        
        # 显示原图的数值（便于理解）
        axes[1, 0].axis('off')
        axes[1, 0].text(0.1, 0.5, '原始图像数值:\n' + str(small_image), 
                       fontfamily='monospace', fontsize=8)
        
        for i, (method, title) in enumerate(zip(padding_methods, titles)):
            # 进行填充（每边填充2个像素）
            padded_image = self.manual_padding(small_image, 2, mode=method)
            
            # 显示填充结果
            axes[0, i+1].imshow(padded_image, cmap='gray')
            axes[0, i+1].set_title(f'{title}结果')
            axes[0, i+1].axis('off')
            
            # 显示数值
            axes[1, i+1].axis('off')
            axes[1, i+1].text(0.1, 0.5, f'{title}数值:\n' + str(padded_image), 
                            fontfamily='monospace', fontsize=6)
        
        plt.tight_layout()
        plt.show()

# ==================== 主程序 ====================

def main():
    """主函数：按顺序执行所有任务"""
    print("=" * 60)
    print("图像处理作业程序")
    print("任务1：高斯滤波核比较")
    print("任务2：手动滤波效果测试")  
    print("任务3：边界填充方式比较")
    print("=" * 60)
    
    # 创建图像处理器对象
    processor = ImageProcessor()
    
    # 任务1：比较高斯滤波核
    print("\n🎯 正在执行任务1：高斯滤波核比较...")
    processor.compare_kernels()
    
    # 任务2：测试滤波效果
    print("\n🎯 正在执行任务2：滤波效果测试...")
    processor.test_filtering()
    
    # 任务3：测试边界填充
    print("\n🎯 正在执行任务3：边界填充测试...")
    processor.test_padding()
    
    print("\n✅ 所有任务完成！")
    print("📊 结果已通过图表显示")
    print("💾 记得将代码上传到GitHub仓库")

# 运行程序
if __name__ == "__main__":
    main()


