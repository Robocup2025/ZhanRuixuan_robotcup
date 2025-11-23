import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 注意这里是 Zen Hei
plt.rcParams['axes.unicode_minus'] = False


def task1_sobel_operator():
    """
    任务1：实现Sobel梯度算子（彩色图像版本）
    """
    # 读取彩色图像
    img = cv2.imread('zyy.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式用于显示
    
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # 对每个颜色通道分别计算梯度
    grad_x_r = cv2.filter2D(img_rgb[:,:,0], cv2.CV_64F, sobel_x)
    grad_x_g = cv2.filter2D(img_rgb[:,:,1], cv2.CV_64F, sobel_x)
    grad_x_b = cv2.filter2D(img_rgb[:,:,2], cv2.CV_64F, sobel_x)
    
    grad_y_r = cv2.filter2D(img_rgb[:,:,0], cv2.CV_64F, sobel_y)
    grad_y_g = cv2.filter2D(img_rgb[:,:,1], cv2.CV_64F, sobel_y)
    grad_y_b = cv2.filter2D(img_rgb[:,:,2], cv2.CV_64F, sobel_y)
    
    # 合并各通道的梯度
    grad_x = np.stack([grad_x_r, grad_x_g, grad_x_b], axis=2)
    grad_y = np.stack([grad_y_r, grad_y_g, grad_y_b], axis=2)
    
    # 计算梯度幅值（对每个通道分别计算，然后取最大值或平均值）
    grad_magnitude_r = np.sqrt(grad_x_r**2 + grad_y_r**2)
    grad_magnitude_g = np.sqrt(grad_x_g**2 + grad_y_g**2)
    grad_magnitude_b = np.sqrt(grad_x_b**2 + grad_y_b**2)
    
    # 方法1：取各通道的最大值作为最终梯度幅值
    grad_magnitude = np.maximum.reduce([grad_magnitude_r, grad_magnitude_g, grad_magnitude_b])
    
    # 方法2：取各通道的平均值作为最终梯度幅值（可选）
    # grad_magnitude = (grad_magnitude_r + grad_magnitude_g + grad_magnitude_b) / 3
    
    # 计算梯度方向（使用幅值最大的通道的方向，或者取平均值）
    # 这里使用红色通道的方向作为代表，也可以采用更复杂的方法
    grad_direction = np.arctan2(grad_y_r, grad_x_r)
    
    # 归一化显示
    grad_magnitude_norm = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 对X和Y方向梯度也进行归一化显示（取各通道的最大值）
    grad_x_magnitude = np.maximum.reduce([np.abs(grad_x_r), np.abs(grad_x_g), np.abs(grad_x_b)])
    grad_y_magnitude = np.maximum.reduce([np.abs(grad_y_r), np.abs(grad_y_g), np.abs(grad_y_b)])
    
    grad_x_norm = cv2.normalize(grad_x_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_y_norm = cv2.normalize(grad_y_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('原始彩色图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(grad_x_norm, cmap='gray')
    plt.title('Sobel X方向梯度')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(grad_y_norm, cmap='gray')
    plt.title('Sobel Y方向梯度')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(grad_magnitude_norm, cmap='gray')
    plt.title('梯度幅值')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task1_sobel_results_color.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grad_magnitude, grad_direction

# 运行函数
if __name__ == "__main__":
    grad_magnitude, grad_direction = task1_sobel_operator()