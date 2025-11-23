import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def task1_sobel_operator():
    """
    任务1：实现Sobel梯度算子
    """
    # 读取图像
    img = cv2.imread('zzz.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # 卷积计算梯度
    grad_x = cv2.filter2D(gray, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, sobel_y)
    
    # 计算梯度幅值和方向
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x)
    
    # 归一化显示
    grad_magnitude_norm = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(grad_x, cmap='gray')
    plt.title('Sobel X方向梯度')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(grad_y, cmap='gray')
    plt.title('Sobel Y方向梯度')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(grad_magnitude_norm, cmap='gray')
    plt.title('梯度幅值')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task1_sobel_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return grad_magnitude, grad_direction

def task2_canny_edge_detector():
    """
    任务2：手动实现Canny边缘检测算法
    """
    # 读取图像
    img = cv2.imread('zzz.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 步骤1: 高斯滤波
    sigma = 1.0
    gray_blur = cv2.GaussianBlur(gray, (5, 5), sigma)
    
    # 步骤2: 使用Sobel计算梯度
    grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    
    # 步骤3: 非极大值抑制
    def non_maximum_suppression(magnitude, direction):
        M, N = magnitude.shape
        suppressed = np.zeros((M, N))
        
        # 将角度量化到0°, 45°, 90°, 135°
        direction = direction % 180
        direction[direction < 0] += 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                # 根据梯度方向确定比较的像素
                if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= direction[i,j] < 67.5:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                elif 67.5 <= direction[i,j] < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= direction[i,j] < 157.5
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                
                # 非极大值抑制
                if magnitude[i,j] >= max(neighbors):
                    suppressed[i,j] = magnitude[i,j]
        
        return suppressed
    
    nms_result = non_maximum_suppression(grad_magnitude, grad_direction)
    
    # 步骤4: 双阈值检测和边缘连接
    def double_threshold_hysteresis(image, low_threshold=50, high_threshold=150):
        M, N = image.shape
        result = np.zeros((M, N))
        
        # 强边缘
        strong_i, strong_j = np.where(image >= high_threshold)
        # 弱边缘
        weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
        
        result[strong_i, strong_j] = 255
        
        # 边缘连接
        for i, j in zip(weak_i, weak_j):
            if np.any(result[i-1:i+2, j-1:j+2] == 255):
                result[i, j] = 255
        
        return result.astype(np.uint8)
    
    edges = double_threshold_hysteresis(nms_result, 30, 100)
    
    # 使用OpenCV的Canny进行比较
    edges_cv2 = cv2.Canny(gray_blur, 30, 100)
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray_blur, cmap='gray')
    plt.title('高斯滤波后')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(grad_magnitude, cmap='gray')
    plt.title('梯度幅值')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(nms_result, cmap='gray')
    plt.title('非极大值抑制后')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(edges, cmap='gray')
    plt.title('手动实现Canny')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(edges_cv2, cmap='gray')
    plt.title('OpenCV Canny')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task2_canny_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return edges

def task3_harris_corner_detector():
    """
    任务3：手动实现Harris角点检测算法
    """
    # 读取图像
    img = cv2.imread('zzz.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # 步骤1: 计算梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 步骤2: 计算M矩阵的元素
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 步骤3: 高斯加权
    window_size = 3
    k = 0.04
    
    Ix2 = cv2.GaussianBlur(Ix2, (window_size, window_size), 1.5)
    Iy2 = cv2.GaussianBlur(Iy2, (window_size, window_size), 1.5)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1.5)
    
    # 步骤4: 计算角点响应函数R
    R = np.zeros_like(gray)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            M = np.array([[Ix2[i,j], Ixy[i,j]],
                         [Ixy[i,j], Iy2[i,j]]])
            # 计算角点响应 R = det(M) - k * trace(M)^2
            det = np.linalg.det(M)
            trace = np.trace(M)
            R[i,j] = det - k * (trace ** 2)
    
    # 步骤5: 非极大值抑制
    def non_max_suppression(R, window_size=3, threshold=0.01):
        corners = []
        R_max = np.max(R)
        
        for i in range(window_size//2, R.shape[0]-window_size//2):
            for j in range(window_size//2, R.shape[1]-window_size//2):
                if R[i,j] > threshold * R_max:
                    # 检查是否是局部最大值
                    local_window = R[i-window_size//2:i+window_size//2+1, 
                                    j-window_size//2:j+window_size//2+1]
                    if R[i,j] == np.max(local_window):
                        corners.append((j, i, R[i,j]))
        
        return corners
    
    corners = non_max_suppression(R, threshold=0.01)
    
    # 可视化结果
    img_display = cv2.imread('zzz.jpg')
    img_with_corners = img_display.copy()
    
    for corner in corners:
        x, y, response = corner
        cv2.circle(img_with_corners, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 使用OpenCV的Harris角点检测进行比较
    img_cv2 = img_display.copy()
    gray_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    gray_cv2 = np.float32(gray_cv2)
    dst = cv2.cornerHarris(gray_cv2, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img_cv2[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    plt.title('手动实现Harris角点')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Harris角点')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task3_harris_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corners

def task4_histogram_equalization():
    """
    任务4：手动实现直方图均衡化
    """
    # 读取图像
    img = cv2.imread('zzz.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 手动实现直方图均衡化
    def manual_histogram_equalization(image):
        # 计算直方图
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # 计算累积分布函数
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]  # 归一化到0-255
        
        # 使用查找表进行映射
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(image.shape)
        
        return equalized.astype(np.uint8)
    
    # 应用直方图均衡化
    equalized_manual = manual_histogram_equalization(gray)
    
    # 使用OpenCV的直方图均衡化进行比较
    equalized_cv2 = cv2.equalizeHist(gray)
    
    # 计算直方图
    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_manual = cv2.calcHist([equalized_manual], [0], None, [256], [0, 256])
    hist_cv2 = cv2.calcHist([equalized_cv2], [0], None, [256], [0, 256])
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 显示图像
    plt.subplot(3, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(equalized_manual, cmap='gray')
    plt.title('手动直方图均衡化')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(equalized_cv2, cmap='gray')
    plt.title('OpenCV直方图均衡化')
    plt.axis('off')
    
    # 显示直方图
    plt.subplot(3, 3, 4)
    plt.plot(hist_original, color='black')
    plt.title('原始直方图')
    plt.xlim([0, 256])
    
    plt.subplot(3, 3, 5)
    plt.plot(hist_manual, color='black')
    plt.title('均衡化后直方图(手动)')
    plt.xlim([0, 256])
    
    plt.subplot(3, 3, 6)
    plt.plot(hist_cv2, color='black')
    plt.title('均衡化后直方图(OpenCV)')
    plt.xlim([0, 256])
    
    # 显示对比结果
    plt.subplot(3, 3, 7)
    plt.imshow(np.hstack([gray, equalized_manual]), cmap='gray')
    plt.title('原始 vs 手动均衡化')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(np.hstack([gray, equalized_cv2]), cmap='gray')
    plt.title('原始 vs OpenCV均衡化')
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    plt.imshow(np.hstack([equalized_manual, equalized_cv2]), cmap='gray')
    plt.title('手动 vs OpenCV均衡化')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('task4_histogram_results.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return equalized_manual, equalized_cv2

def main():
    """
    主函数：依次执行四个任务
    """
    print("开始执行homework3的四个任务...")
    
    print("\n=== 任务1: Sobel梯度算子 ===")
    task1_sobel_operator()
    
    print("\n=== 任务2: Canny边缘检测 ===")
    task2_canny_edge_detector()
    
    print("\n=== 任务3: Harris角点检测 ===")
    task3_harris_corner_detector()
    
    print("\n=== 任务4: 直方图均衡化 ===")
    task4_histogram_equalization()
    
    print("\n所有任务已完成！结果图像已保存。")

if __name__ == "__main__":
    main()
