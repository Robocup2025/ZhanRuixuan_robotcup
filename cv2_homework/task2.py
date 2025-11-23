import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 注意这里是 Zen Hei
plt.rcParams['axes.unicode_minus'] = False


def complete_canny_detection_with_steps(image_path, sigma=1.0, low_threshold=50, high_threshold=150):
    """
    完整的Canny边缘检测，包含所有中间步骤的可视化
    """
    # 读取图像
    img = cv2.imread('zyy.jpg')
    if img is None:
        print("无法读取图像，请检查图像路径")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 高斯滤波
    blurred = cv2.GaussianBlur(gray, (5, 5), sigma)
    
    # 2. 计算梯度幅度和方向（梯度图）
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅度图
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # 梯度方向图
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # 3. 非极大值抑制 (NMS)
    def non_maximum_suppression(magnitude, direction):
        M, N = magnitude.shape
        suppressed = np.zeros((M, N))
        
        # 将角度转换为度
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255
                    r = 255
                    
                    # 角度0°
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = magnitude[i, j+1]
                        r = magnitude[i, j-1]
                    # 角度45°
                    elif 22.5 <= angle[i,j] < 67.5:
                        q = magnitude[i+1, j-1]
                        r = magnitude[i-1, j+1]
                    # 角度90°
                    elif 67.5 <= angle[i,j] < 112.5:
                        q = magnitude[i+1, j]
                        r = magnitude[i-1, j]
                    # 角度135°
                    elif 112.5 <= angle[i,j] < 157.5:
                        q = magnitude[i-1, j-1]
                        r = magnitude[i+1, j+1]

                    if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                        suppressed[i,j] = magnitude[i,j]
                    else:
                        suppressed[i,j] = 0

                except IndexError:
                    pass
        
        return suppressed
    
    nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # 4. 双阈值检测和边缘连接
    def double_threshold_hysteresis(image, low_threshold, high_threshold):
        M, N = image.shape
        result = np.zeros((M, N), dtype=np.uint8)
        
        # 定义强边缘和弱边缘
        strong = 255
        weak = 50
        
        # 找到强边缘和弱边缘位置
        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
        
        # 设置强边缘
        result[strong_i, strong_j] = strong
        # 设置弱边缘
        result[weak_i, weak_j] = weak
        
        # 边缘连接 - 如果弱边缘与强边缘相邻，则提升为强边缘
        for i in range(1, M-1):
            for j in range(1, N-1):
                if result[i,j] == weak:
                    if ((result[i+1, j-1] == strong) or (result[i+1, j] == strong) or 
                        (result[i+1, j+1] == strong) or (result[i, j-1] == strong) or 
                        (result[i, j+1] == strong) or (result[i-1, j-1] == strong) or 
                        (result[i-1, j] == strong) or (result[i-1, j+1] == strong)):
                        result[i,j] = strong
                    else:
                        result[i,j] = 0
        
        return result
    
    edges_connected = double_threshold_hysteresis(nms_result, low_threshold, high_threshold)
    
    # 5. 使用OpenCV的Canny函数进行对比
    opencv_edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 可视化所有步骤
    plt.figure(figsize=(16, 10))
    
    # 原图和灰度图
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('灰度图')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title('高斯滤波后')
    plt.axis('off')
    
    # 梯度图
    plt.subplot(2, 4, 4)
    plt.imshow(gradient_magnitude, cmap='hot')
    plt.title('梯度幅度图')
    plt.colorbar()
    plt.axis('off')
    
    # NMS图
    plt.subplot(2, 4, 5)
    plt.imshow(nms_result, cmap='hot')
    plt.title('非极大值抑制(NMS)')
    plt.colorbar()
    plt.axis('off')
    
    # 边缘连接图
    plt.subplot(2, 4, 6)
    plt.imshow(edges_connected, cmap='gray')
    plt.title('边缘连接图')
    plt.axis('off')
    
    # OpenCV Canny结果
    plt.subplot(2, 4, 7)
    plt.imshow(opencv_edges, cmap='gray')
    plt.title('OpenCV Canny结果')
    plt.axis('off')
    
    # 最终对比
    plt.subplot(2, 4, 8)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(edges_connected, cmap='jet', alpha=0.5)
    plt.title('边缘叠加图')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original': gray,
        'blurred': blurred,
        'gradient_magnitude': gradient_magnitude,
        'nms_result': nms_result,
        'edges_connected': edges_connected,
        'opencv_edges': opencv_edges
    }

# 使用示例
if __name__ == "__main__":
    # 请将下面的路径替换为您的图像路径
    image_path = "your_image.jpg"  # 替换为您的图像路径
    
    # 执行完整的Canny边缘检测
    results = complete_canny_detection_with_steps(image_path)
    
    if results is not None:
        print("Canny边缘检测完成！")
        print("包含以下步骤的结果：")
        print("1. 梯度图 (gradient_magnitude)")
        print("2. 非极大值抑制图 (nms_result)") 
        print("3. 边缘连接图 (edges_connected)")
        print("4. OpenCV Canny结果对比")
