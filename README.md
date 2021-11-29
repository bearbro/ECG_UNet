# Fork from https://github.com/Aiwiscal/ECG_UNet
代码和模型
# 我写的代码：
* data_preproc_ccdd.py   处理数据
* data_preproc_tianchi.py 处理数据
* train_my.py 训练模型
* test_a_sig_my.py 可视化预测结果
* Unet.py 添加crf层得到Unet\_crf 模型


# 运行前需要做的事
* 更改数据路径（ccdd、天池）

# 运行顺序
1. python  data_preproc_ccdd.py
2. python train_my.py
3. python test_a_sig_my.py
