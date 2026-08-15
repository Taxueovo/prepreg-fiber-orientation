# E01 预训练初始化对照

目标：证明自监督 DINOv3 初始化是否在固定数据协议下优于随机初始化、ImageNet 监督初始化和 DINOv2。当前项目只保存了 DINOv3 原型，尚不能证明预训练贡献。

直接运行示例：`python thesis_experiment_packages/01_pretraining_controls.py --variant dinov3 --weights /绝对路径/dinov3.pth`。随机初始化则执行 `--variant random_init` 且不传权重。对四种初始化各训练 5 个随机种子；除初始化权重外，输入、训练集、验证集、训练轮数、早停、FFT/注意力配置必须相同。测试集不得用于选学习率、环带参数或最佳 epoch。

每次运行将逐图块预测按 `source_image` 聚合为逐母图预测，写入 `results/E01_pretraining/parent_predictions_template.csv` 的同名正式文件。论文表 4.4 报告每种初始化的母图级 MAE、RMSE、95% cluster-bootstrap CI，以及相对随机初始化的配对误差差值。若区间跨零，应报告无稳定差异。
