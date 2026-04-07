# 桌面版说明

## 运行桌面版

1. 安装依赖

```powershell
$mirrors = @("https://pypi.tuna.tsinghua.edu.cn/simple", "https://mirrors.aliyun.com/pypi/simple", "https://pypi.mirrors.ustc.edu.cn/simple", "https://repo.huaweicloud.com/repository/pypi/simple", "https://pypi.org/simple"); foreach ($m in $mirrors) { $h = ([Uri]$m).Host; Write-Host "尝试源: $m"; .\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt -i $m --trusted-host $h --retries 1 --timeout 10; if ($LASTEXITCODE -eq 0) { break } }
```

如果你直接执行 `build_desktop.ps1`，脚本也会自动按上述顺序回退镜像站。
打包脚本会自动安装 `requirements-desktop.txt` 和 `backend/requirements.txt`。

2. 启动桌面应用

```powershell
.\.venv\Scripts\python.exe .\desktop_app.py
```

桌面应用会在进程内自动启动 Flask 后端，不再依赖 Web 前端。

## 打包 EXE

执行脚本：

```powershell
.\build_desktop.ps1
```

输出目录：

- `dist\ZhishiExeDesktop.exe`

## 创建桌面快捷方式

执行：

```powershell
.\create_desktop_shortcut.ps1
```

脚本会自动：

- 复制 `dist\ZhishiExeDesktop.exe` 到桌面目录 `ZhishiExeDesktop`
- 在桌面创建快捷方式 `ZhishiExeDesktop.lnk`

## 当前工作流

- 文件上传与解析
- 知识抽取与图谱构建
- 图谱结构数据查看
- 智能问答与历史管理
