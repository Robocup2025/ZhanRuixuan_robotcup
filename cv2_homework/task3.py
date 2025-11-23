import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 注意这里是 Zen Hei
plt.rcParams['axes.unicode_minus'] = False


def harris_corner_detection(image_path, sigma=1.0, window_size=3, alpha=0.04, threshold=0.01, nms_size=3):
    """
    Harris角点检测完整实现
    """
    # 1. 读取图像并转换为灰度图
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查图像路径")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # 2. 高斯滤波（步骤1）
    gray_blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # 3. 计算梯度Ix, Iy（步骤2）
    Ix = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 4. 计算Ix², Iy², IxIy（步骤3）
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    # 5. 对梯度乘积进行高斯滤波（步骤4）
    Ix2_blurred = cv2.GaussianBlur(Ix2, (window_size, window_size), 1.5)
    Iy2_blurred = cv2.GaussianBlur(Iy2, (window_size, window_size), 1.5)
    Ixy_blurred = cv2.GaussianBlur(Ixy, (window_size, window_size), 1.5)
    
    # 6. 构造二阶矩矩阵M并计算响应函数R（步骤5-6）
    height, width = gray.shape
    R = np.zeros_like(gray)
    
    for i in range(height):
        for j in range(width):
            # 构造二阶矩矩阵M
            M = np.array([[Ix2_blurred[i, j], Ixy_blurred[i, j]],
                         [Ixy_blurred[i, j], Iy2_blurred[i, j]]])
            
            # 计算Harris响应函数R = det(M) - α * trace(M)²
            det_M = np.linalg.det(M)#这个函数用于计算矩阵行列式
            trace_M = np.trace(M)#计算矩阵的迹（对角线元素之和）
            R[i, j] = det_M - alpha * (trace_M ** 2)
    
    # 7. 阈值处理（步骤7）
    corners = R > (threshold * R.max())
    
    # 8. 非极大值抑制（步骤8）
    def non_maximum_suppression(response, size=3):
        suppressed = response.copy()
        height, width = response.shape
        
        for i in range(size//2, height - size//2):
            for j in range(size//2, width - size//2):
                # 获取局部邻域
                neighborhood = response[i-size//2:i+size//2+1, j-size//2:j+size//2+1]
                
                # 如果当前点不是局部最大值，则抑制
                if response[i, j] != np.max(neighborhood):
                    suppressed[i, j] = 0
        
        return suppressed
    
    R_suppressed = non_maximum_suppression(R, nms_size)
    final_corners = R_suppressed > (threshold * R_suppressed.max())
    
    # 使用OpenCV库函数进行对比
    opencv_corners = cv2.cornerHarris(gray_blurred, window_size, 3, alpha)
    opencv_corners = cv2.dilate(opencv_corners, None)#对图像进行膨胀操作，使角点标记更明显
    opencv_result = img.copy()
    opencv_result[opencv_corners > 0.01 * opencv_corners.max()] = [0, 0, 255]
    '''
    np.max(neighborhood)  # 返回数组最大值
np.argmax(corner_counts)  # 返回最大值索引
corners_coords = np.where(results['corners_before_nms'])返回满足条件的数组元素的坐标
opencv_result[opencv_corners > 0.01 * opencv_corners.max()] = [0, 0, 255]这是找出所有响应值大于
最大响应值1%的区域并把他们弄成红色
    '''
    
    return {
        'original': img,
        'gray': gray,
        'Ix': Ix,
        'Iy': Iy,
        'Ix2': Ix2,
        'Iy2': Iy2,
        'Ixy': Ixy,
        'response': R,
        'corners_before_nms': corners,
        'corners_after_nms': final_corners,
        'opencv_result': opencv_result
    }

def analyze_window_size_impact(image_path):
    """
    分析窗口大小对角点检测的影响
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    window_sizes = [3, 5, 7, 9, 11]
    results = []
    
    for window_size in window_sizes:
        # 使用OpenCV的Harris角点检测
        harris_response = cv2.cornerHarris(gray, window_size, 3, 0.04)
        harris_response = cv2.dilate(harris_response, None)
        
        # 标记角点
        result_img = img.copy()
        result_img[harris_response > 0.01 * harris_response.max()] = [0, 0, 255]
        
        # 统计角点数量
        corner_count = np.sum(harris_response > 0.01 * harris_response.max())
        '''
corner_count = np.sum(harris_response > threshold)
计算数组元素的和，这里用于统计角点数量
        '''
        results.append({
            'window_size': window_size,
            'image': result_img,
            'corner_count': corner_count,
            'response': harris_response
        })
    
    return results

def visualize_results(results, window_analysis_results):
    """
    可视化所有结果
    """
    # 主结果可视化
    fig = plt.figure(figsize=(20, 15))
    
    # 原始图像和梯度
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(results['Ix'], cmap='hot')
    plt.title('梯度Ix')
    plt.colorbar()#显示颜色条
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(results['Iy'], cmap='hot')
    plt.title('梯度Iy')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(results['Ix2'], cmap='hot')
    plt.title('Ix²')
    plt.colorbar()
    plt.axis('off')
    
    # 梯度乘积和响应函数
    plt.subplot(3, 4, 5)
    plt.imshow(results['Iy2'], cmap='hot')
    plt.title('Iy²')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(results['Ixy'], cmap='hot')
    plt.title('IxIy')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(results['response'], cmap='hot')
    plt.title('Harris响应函数R')
    plt.colorbar()
    plt.axis('off')
    
    # 角点检测结果
    result_before_nms = results['original'].copy()
    corners_coords = np.where(results['corners_before_nms'])
    result_before_nms[corners_coords] = [0, 0, 255]
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(result_before_nms, cv2.COLOR_BGR2RGB))
    plt.title(f'阈值处理后角点\n({len(corners_coords[0])}个角点)')
    plt.axis('off')
    
    result_after_nms = results['original'].copy()
    corners_coords_nms = np.where(results['corners_after_nms'])
    result_after_nms[corners_coords_nms] = [0, 0, 255]
    
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(result_after_nms, cv2.COLOR_BGR2RGB))
    plt.title(f'NMS后角点\n({len(corners_coords_nms[0])}个角点)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(cv2.cvtColor(results['opencv_result'], cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Harris结果')
    plt.axis('off')
    
    # 窗口大小影响分析
    window_sizes = [r['window_size'] for r in window_analysis_results]
    corner_counts = [r['corner_count'] for r in window_analysis_results]
    
    plt.subplot(3, 4, 11)
    plt.plot(window_sizes, corner_counts, 'bo-', linewidth=2, markersize=8)#绘制折线图
    plt.xlabel('窗口大小')
    plt.ylabel('检测到的角点数量')
    plt.title('窗口大小对角点检测的影响')
    plt.grid(True)
    
    plt.subplot(3, 4, 12)
    best_idx = np.argmax(corner_counts)
    plt.imshow(cv2.cvtColor(window_analysis_results[best_idx]['image'], cv2.COLOR_BGR2RGB))
    plt.title(f'最佳窗口大小: {window_sizes[best_idx]}\n({corner_counts[best_idx]}个角点)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()#显示所有图表
    
    # 打印分析结果
    print("=== 窗口大小影响分析 ===")
    for i, result in enumerate(window_analysis_results):
        print(f"窗口大小 {result['window_size']}: 检测到 {result['corner_count']} 个角点")

# 主程序
if __name__ == "__main__":
    image_path = "zyy.jpg"  # 请确保图片在同一文件夹
    
    print("开始Harris角点检测...")
    
    # 执行Harris角点检测
    results = harris_corner_detection(image_path, 
                                    sigma=1.0, 
                                    window_size=3, 
                                    alpha=0.04, 
                                    threshold=0.01)
    
    if results is not None:
        # 分析窗口大小影响
        print("分析窗口参数影响...")
        window_analysis_results = analyze_window_size_impact(image_path)
        
        # 可视化结果
        visualize_results(results, window_analysis_results)
        
        print("Harris角点检测完成！")
        print(f"自定义实现检测到角点: {np.sum(results['corners_after_nms'])} 个")
    else:
        print("图像读取失败，请检查图像路径")
