def pyplot_中文():
    import sys
    if sys.platform.startswith("linux"):
        print("linux")
    elif sys.platform == "darwin":
        # MAC OS X
        try:
            import seaborn as sns
            sns.set(font="Arial Unicode MS")  # "DFKai-SB"
            print("Initiated Seaborn font")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
            plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
            plt.rcParams['axes.unicode_minus'] = False
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")

    elif sys.platform == "win32":
        # Windows (either 32-bit or 64-bit)
        try:
            import seaborn as sns
            sns.set(font="sans-serif")  # "DFKai-SB"
            print("Initiated Seaborn font ")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 換成中文的字體
            plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決seaborn座標軸亂碼問題）
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")