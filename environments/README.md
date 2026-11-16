# Environments

`retailvision-seg` 和 `retailvision-reid` 为独立 Conda 环境。Python 3.10 是官方 MMCV cu121/torch2.1 预编译 wheel 的匹配选择；实际 freeze 优先于可浮动安装脚本。

初次安装执行 install-seg.sh（会创建新环境，已有环境不要重复创建）。源码与权重先通过 tools/download_sources.sh 拉取。后续复现使用 conda explicit 文件创建 Python 环境，再 pip install -r 对应 freeze，CUDA torch 包使用 --extra-index-url https://download.pytorch.org/whl/cu121。editable 路径需替换为当前 checkout 的绝对路径，源码 commit 记录在 manifests。

下载完成后保留本地 pip wheel 缓存或制作离线 wheelhouse：联网机器 `pip download -r ... --extra-index-url ... -d wheelhouse`，离线 `pip install --no-index --find-links wheelhouse ...`。此 wheelhouse 未在本次额外复制，以免重复占用磁盘。运行时离线验证与离线全新安装是不同测试：本次验证本地权重运行，未声称完成空气隔离的新机器安装。

`licenses-installed.json` 是实际 Python 包元数据，不代替权重许可核对；前端锁定 package-lock.json。

部署使用独立的 `retailvision-deploy` 环境。`install-deploy.sh` 从 segmentation 环境创建新副本，固定 ONNX / ORT CUDA 12 / TensorRT 10.0.1 和 CUDA 12.4.1 补充组件；不升级训练环境的 PyTorch。实际部署快照为 `deploy-freeze.txt` 和 `deploy-conda-explicit.txt`。CUDA 动态库激活配置为 `deploy-cuda-libs.sh`。安装、导出和验收说明见 [DEPLOYMENT.md](../docs/DEPLOYMENT.md)。

部署快照中的 ORT 1.18.0 仍须通过 DEPLOYMENT.md 指定的官方 CUDA 12 feed 安装，同版本的默认 PyPI wheel 不适用于本部署组合。
