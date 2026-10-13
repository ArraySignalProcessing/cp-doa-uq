# 统一实验环境

项目使用现有`code/.venv/`，依赖统一记录于`code/requirements.txt`。
不为论文新版本复制环境。所有命令从项目根目录执行。

## 创建环境（仅在现有环境不可用时）

```bash
python3.11 -m venv code/.venv
source code/.venv/bin/activate
python -m pip install -r code/requirements.txt
```

## 测试与实验

先由用户运行数学、协议、失败分母和检查点恢复测试：

```bash
source code/.venv/bin/activate
python -m unittest discover -s code/tests -v
```

新实验统一使用`code/run_paper.py`，顺序为`calibration`、`boundary`、`summary`、`plot`。
正式校准固定50次Bootstrap重拟合，不再运行独立预算预实验。
具体命令和协议见项目根目录`PROJECT.md`。当前先检查讨论实验章设计，不启动正式实验。

长任务支持检查点续跑和终端进度。每阶段记录实际依赖版本、CPU架构、平台与线程环境设置。
续跑须使用相同代码、配置、环境和线程设置；不要在正式阶段中途升级依赖或改变线程数。

## 结果边界

新结果与图表位于`data/paper_abc/`。绘图入口不允许输出到论文目录。
旧正式数据、失败试验和论文图表本次不改写。结果检查和替换论文图表分别确认。
助手不代为运行测试、实验、绘图或论文编译。
