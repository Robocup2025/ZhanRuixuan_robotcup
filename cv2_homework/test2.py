import matplotlib.pyplot as plt

# 使用正确的字体名称
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 注意这里是 Zen Hei
plt.rcParams['axes.unicode_minus'] = False

# 测试代码
plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('中文标题测试 - 现在应该能显示了！')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.show()