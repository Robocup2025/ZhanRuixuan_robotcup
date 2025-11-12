import numpy as np
import cv2
import matplotlib.pyplot as plt

class GaosiPic():
    def __init__(self):
        pass
    def loadimage(self):
        imagepath="zyy.jpg"
        color_image=cv2.imread(imagepath)
        #以下转换为RGB格式
        color_image=cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        gray_image=cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
        gray_image=color_image
        return color_image,gray_image
    def My_kernel(self,size,sigma):#自己进行高斯滤波操作COLOR_RGB2GRAY
        mykernel=np.zeros((size,size))
        center=size//2
        for i in range(size):
            for j in range(size):
                x=i-center
                y=j-center
                gxy=(1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
                mykernel[i,j]=gxy
        mykernel=mykernel/np.sum(mykernel)
        return mykernel
    def manual_filter(self,image,size,sigma):
        kernel=self.My_kernel(size,sigma)
        kernel_h,kernel_w=kernel.shape
        pad_h=kernel_h//2#填充高度
        pad_w=kernel_w//2#填充宽度
        # 下面对图像进行边界填充
        padded_image=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),mode='reflect')
        #创建结果图像
        result=np.zeros_like(image,dtype=np.float32)
        #对图像进行滤波操作
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                
                region=padded_image[i:i+kernel_h,j:j+kernel_w]
                #以上是取出像素周围区域
                result[i,j]=np.sum(region*kernel)
        # 确保像素值在0-255之间
        return np.clip(result,0,255).astype(np.uint8)
    
    def test_filter(self):
        kernals=[(31,3.0),(51,9.0),(71,22.0)]
        color_image,gray_image=self.loadimage()
        fig,axes=plt.subplots(1,4,figsize=(16,8))
        axes[0].imshow(gray_image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        for i,(size,sigma) in enumerate(kernals):
            image=self.manual_filter(gray_image,size,sigma)
            axes[i+1].imshow(image,cmap='gray')
            axes[i+1].set_title(f"size*size={size}*{size},sigma={sigma}")
            axes[i+1].axis('off')
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    a=GaosiPic()
    a.test_filter()




            

    

            