import matplotlib.font_manager as fm
# 获取所有可用字体，并检查中文字体
fonts = [f.name for f in fm.fontManager.ttflist if 'wenquan' in f.name.lower()]
print("Matplotlib中的文泉驿字体:", fonts)

# 或者列出所有中文字体
all_chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'chinese' in f.name.lower() or 'chinese' in f.fname.lower()]
print("Matplotlib中的中文字体:", all_chinese_fonts)