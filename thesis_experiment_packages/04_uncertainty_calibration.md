# E04 回归不确定性与选择性预测

当前 r=0.129 只说明方差输出存在很弱的风险排序信号，不能称为“校准良好”。本包要求使用未参与训练与模型选择的校准/测试母图，保存 `parent_id,y_true_deg,y_pred_deg,pred_std_deg`。

先生成模板：`python thesis_experiment_packages/04_uncertainty_calibration.py`；再分析：`python thesis_experiment_packages/04_uncertainty_calibration.py --predictions 你的逐母图预测.csv`。脚本会输出 MAE、RMSE、Spearman 排序相关、50/80/90/95% PICP 与95% MPIW。PICP 是覆盖率，越接近名义覆盖率越好；MPIW 要与覆盖率一起解释，不能只追求很宽的区间。

论文第 5.2 节还应画风险—覆盖率曲线：按预测标准差由小到大保留 50%--100% 样本，分别计算 MAE。若拒绝高风险样本不能显著降低误差，则不应主张人工复核机制有效。
