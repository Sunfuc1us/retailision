# 实际问题记录

1. 环境创建尚未结束时提前执行 conda run 安装：EnvironmentLocationNotFound。等待 conda create 成功后执行原计划安装；没有更换版本。
2. 文档生成辅助脚本缺少 json 导入，已修正后重新执行。项目源码未受影响。
3. 首次浏览器测试在 Chromium 下载完成前执行，报 executable missing；等待下载完成后重跑，两项通过。
4. Fixture 转换已成功，但生成摘要时重复传入 fixture 键导致 TypeError；保留已生成数据，单独修复报告，不覆盖重做数据。
5. Conda 默认读取用户 site-packages，已为两个环境持久设置 PYTHONNOUSERSITE=1，补齐环境内包并测试 sys.path。
6. MMEngine 强制依赖标准 OpenCV，原计划 headless 会导致重复 cv2 包，停止依赖解析并统一 seg 使用 opencv-python<4.12；reid 仅 headless。
7. MMDetection editable build 隔离环境无 torch，使用 --no-build-isolation 构建已安装兼容 torch 的环境，没有变更版本。
8. Recognition resume 将 CUDA RNG ByteTensor 映射至 GPU 后不能 set_state，已明确转换回 CPU 并重跑断点恢复。
9. torch2.1 cpp_extension依赖pkg_resources，新setuptools84已移除；根据实际traceback将seg环境setuptools锁定80.9.0，构建成功。
10. conda run python -未透传测试脚本stdin；改用可复现tools/verify_training_api.py文件执行，没有将空日志算作成功。
11. 训练API实验写入completed后Conda子进程尚在清理，立即启动下一任务返回422；验证工具等待该进程实际退出，再验证第二任务停止。
