# E08 推理效率与系统节拍

目标：报告真实端到端性能，而非只报模型前向时间。运行 `python thesis_experiment_packages/08_efficiency_latency.py` 生成模板；使用固定硬件、软件版本、batch=1 和预热后的重复测量，记录读取、预处理、推理、结果写出以及端到端总时间。

填完 CSV 后运行 `python thesis_experiment_packages/08_efficiency_latency.py --samples latency.csv`，输出平均值、P50、P95、吞吐量和峰值显存。采集、传输与人工复核如未计入端到端样本，必须在论文中单独列出，不能以模型延迟代替产线节拍。

第 5.4 节报告硬件、软件、图像尺寸、测量次数、延迟分布、吞吐量、显存、参数量和 FLOPs；如没有真实产线集成，不应外推“满足在线部署”。
