# 本地部署

Ubuntu 22.04 / RTX 4090 / Driver 580.178.04（驱动报告 CUDA 13.0 为支持上限，不等于已安装 toolkit）。未安装 nvcc；MMCV 使用官方预编译 CUDA wheel，训练和推理不需编译 CUDA 源码。

兼容性决定：MMDetection 3.3.0 官方约束 mmcv>=2.0,<2.2 与 mmengine<1。采用已存在的 mmcv 2.1.0 cp310 cu121 torch2.1 wheel，配 torch 2.1.2 + torchvision .16.2；RTX4090 支持 CUDA12.1，驱动支持该 runtime。采用旧框架验证组合有维护成本，未盲目将当前 PyTorch 混入 MMCV。

完整实际 pip freeze 与 conda explicit 见 environments/。安装日志和失败记录在 logs/。只操作独立 conda 环境，不修改全局 Python 或驱动。DINO 不安装上游整个预训练依赖集（xformers/cuML 非本任务需要），使用 PyTorch 原生 attention。

启动命令见 README。前端可由 FastAPI 托管 dist；开发 Vite 代理到 8018。API 默认仅本机使用，不作为公网服务。没有数据库，备份 catalog、datasets、models、work_dirs、results 即保存所有研发产物。
