from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

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

def ML_read_excel(fileName,featuresCol,labelCol):
    df=pd.read_excel(fileName)        #讀取 pandas資料
    print(df.columns)                    #印出所有列
    x=df[featuresCol]          #設定x的資料
    print(x)                                                                                       #印出x的資料
    y=df[labelCol]                                                                               #設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy (參考Day34-524)
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列(參考Day37-573)
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return featuresCol,labelCol,train_x, test_x, train_y, test_y
    # #印出y的資料

def ML_分類_DecisionTree(iris_X_train, iris_X_test, iris_y_train, iris_y_test):
    clf = tree.DecisionTreeClassifier()#初始化決策樹
    clf = clf.fit(iris_X_train,iris_y_train)  #透過資料去訓練
    x1 = clf.predict(iris_X_test)  # 透過資料去訓練
    print("印出預測答案",x1)#印出預測答案
    print("真正的答案",iris_y_test)#印出真正的答案
    tree.export_graphviz(clf,out_file='tree.dot')#輸出城tree.dot的檔案


    # 畫出決策樹
    fig = plt.figure()
    tree.plot_tree(clf,
                       filled=True)
    fig.savefig("decistion_tree1.png")
    plt.show()