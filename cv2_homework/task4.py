import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用更常见的中文字体
plt.rcParams['axes.unicode_minus'] = False

def manual_histogram_equalization(image):
    """
    手动实现直方图均衡化
    """
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])#image.flatten()将多维数组展平为一维数组
    
    # 计算累积分布函数
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # 归一化到0-255
    
    # 使用查找表进行映射
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(image.shape)#将一维数组重新变成原图像形状
    
    return equalized.astype(np.uint8)

def histogram_equalization_demo():
    """
    直方图均衡化演示函数
    """
    # 读取图像 - 如果文件不存在，创建一个示例图像
    try:
        img = cv2.imread('zyy.jpg')
        if img is None:
            raise FileNotFoundError("图像文件未找到，创建示例图像...")
    except:
        # 创建一个示例图像（渐变图像）
        print("创建示例图像用于演示...")
        img = np.zeros((300, 400), dtype=np.uint8)
        for i in range(300):
            img[i, :] = i * 255 // 300
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"图像尺寸: {gray.shape}")
    print(f"像素范围: [{gray.min()}, {gray.max()}]")
    
    # 应用直方图均衡化
    equalized_manual = manual_histogram_equalization(gray)
    equalized_cv2 = cv2.equalizeHist(gray)#opencv内置直方图均值化函数
    
    # 计算直方图
    hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_manual = cv2.calcHist([equalized_manual], [0], None, [256], [0, 256])
    hist_cv2 = cv2.calcHist([equalized_cv2], [0], None, [256], [0, 256])
    '''
    作用: 计算图像直方图
参数:

    [gray]: 输入图像列表

    [0]: 通道索引(灰度图为0)

    None: 掩码

    [256]: 直方图大小(bin数量)

    [0, 256]: 像素值范围
    '''
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
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 5)
    plt.plot(hist_manual, color='black')
    plt.title('均衡化后直方图(手动)')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 6)
    plt.plot(hist_cv2, color='black')
    plt.title('均衡化后直方图(OpenCV)')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)#配置网格线
    
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
    plt.savefig('histogram_equalization_results.jpg', dpi=300, bbox_inches='tight')#将图像保存
    plt.show()
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"原始图像 - 均值: {gray.mean():.2f}, 标准差: {gray.std():.2f}")#mean是用来返回平均值的函数
    print(f"手动均衡化 - 均值: {equalized_manual.mean():.2f}, 标准差: {equalized_manual.std():.2f}")
    print(f"OpenCV均衡化 - 均值: {equalized_cv2.mean():.2f}, 标准差: {equalized_cv2.std():.2f}")
    # std用来返回数组标准差
    # 检查两种方法的结果差异
    diff = cv2.absdiff(equalized_manual, equalized_cv2)#作用: 计算两个图像的绝对差异
    print(f"\n手动与OpenCV结果差异 - 最大差异: {diff.max()}, 平均差异: {diff.mean():.4f}")
    
    return equalized_manual, equalized_cv2

def process_custom_image(image_path):
    """
    处理自定义图像
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_manual = manual_histogram_equalization(gray)
        equalized_cv2 = cv2.equalizeHist(gray)
        
        # 显示简单对比
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(equalized_manual, cmap='gray')
        plt.title('手动均衡化')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(equalized_cv2, cmap='gray')
        plt.title('OpenCV均衡化')
        plt.axis('off')#隐藏坐标轴
        
        plt.tight_layout()
        plt.show()
        
        return equalized_manual, equalized_cv2
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

if __name__ == "__main__":
    print("=== 直方图均衡化演示 ===")
    print("1. 使用内置示例图像")
    print("2. 处理自定义图像")
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == "2":
        image_path = input("请输入图像路径: ").strip()
        process_custom_image(image_path)
    else:
        # 运行主演示
        manual_result, cv2_result = histogram_equalization_demo()
        
        # 保存结果
        cv2.imwrite('manual_equalization.jpg', manual_result)
        cv2.imwrite('opencv_equalization.jpg', cv2_result)
        print("\n结果已保存为:")
        print("- histogram_equalization_results.jpg (对比图)")
        print("- manual_equalization.jpg (手动均衡化结果)")
        print("- opencv_equalization.jpg (OpenCV均衡化结果)")