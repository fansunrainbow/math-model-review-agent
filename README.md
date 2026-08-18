# 数学建模论文评审 Agent

一个面向数学建模竞赛论文的“指导型评审 Agent”。项目对整篇论文进行章节评审、跨章节事实核验、优秀论文知识库检索与最终报告审计，目标是帮助作者定位和排序修改任务，而不是直接代写论文。

## 主要能力

- 章节级与全文级审查，避免把“本节没写”误判成“全文没写”；
- 基于 `known_facts`、`removed_claims` 和 `forbidden_phrases` 的事实锁定；
- 合并重复问题，并按 P0 / P1 / P2 给出修改优先级；
- 从本地 Chroma 知识库检索优秀论文中的同类写法；
- 对最终报告执行事实越权审计；
- 对 PDF 转 Markdown 产生的公式、表格与 OCR 伪影采取保守处理。

## 当前状态：V3.0.6

当前代码层版本为 `3.0.6`，JSON 运行时补丁版本为 `1.0`。正式入口 `review_agent_main.py` 与版本快照 `review_agent_main_v3_0_6.py` 内容一致。

V3.0.6 已通过 Python 语法检查、Parser Artifact Guard 的 test4 fixture 重放和通用边界测试，但尚未依据用户本机重新运行的真实 test4 输出完成最终冻结。不要把当前版本描述为已冻结。

下一项验收工作是重新运行 `test4.pdf`，检查：

- I01、I02 只出现在“原 PDF 待核验项”；
- I03 的正式问题只保留“HHI 定义表述不清”，且不再包含公式(32)、2900、5220 或无关章节；
- I04、I05 仍为正式 P1；
- `06_report_fact_audit_v3.json` 通过。

详细开发背景见 `math_model_review_agent_v3_0_6_context.md`。

## 主要入口与文件

| 文件 | 用途 |
| --- | --- |
| `review_agent_main.py` | 当前正式主程序（bundled 单文件版本） |
| `review_pdf_main.py` | PDF / Markdown / TXT 包装入口；PDF 由 MinerU 解析 |
| `review_agent_main_v3_0_6.py` | V3.0.6 版本快照 |
| `search_all_v3.py` | Chroma 检索与本地 embedding rerank |
| `split_paper.py` | 论文结构切分 |
| `rubric_v1.json` | 评审 rubric |
| `experience_library_v1_frozen.jsonl` | 冻结的经验库 |
| `batch_parse.py`、`build_all_structured.py`、`make_chunks.py`、`build_all_vector_db.py` | 知识库构建脚本 |

若只希望上传可维护源码，建议保留入口、版本快照、构建脚本、rubric、经验库和本文档；本地论文语料、解析结果、模型权重及向量数据库不应直接提交。

## 安装与环境准备（Windows + PowerShell）

以下命令均在 PowerShell 中执行。项目当前按 Windows 和 Python 3.12 开发；建议优先使用 64 位 Python 3.12，避免不同 Python 大版本造成 MinerU、PyTorch 或 embedding 依赖不兼容。

### 1. 前置条件

先安装并确认以下工具：

- Git（若使用 `git clone`）；
- 64 位 Python 3.12，并确保 Python Launcher `py` 可用；
- PowerShell 5.1 或 PowerShell 7；
- 合法获得的本地 embedding 模型、Chroma 数据库和论文语料；
- 有效且已妥善保管的 DeepSeek API Key。

检查 Python：

```powershell
py -3.12 --version
py -0p
```

第一条应显示 `Python 3.12.x`。若系统找不到 `py`，可在确认 `python --version` 指向 Python 3.12 后，将后续的 `py -3.12` 换成 `python`。

### 2. 获取项目并进入目录

从 GitHub 克隆时，将占位地址替换为项目的真实仓库地址：

```powershell
Set-Location D:\
git clone <仓库地址> math_model_kb
Set-Location D:\math_model_kb
```

如果下载的是 ZIP，请先解压，再使用带引号的完整路径进入项目。例如：

```powershell
Set-Location "D:\math_model_kb"
```

当前 `review_pdf_main.py`、`search_all_v3.py` 等脚本仍使用 `D:/math_model_kb` 绝对路径。最省事的做法是把项目放在 `D:\math_model_kb`；若使用其他目录，必须先将源码中的这些绝对路径改为实际路径或改造成配置项。

确认入口文件存在：

```powershell
Get-Item .\review_agent_main.py, .\review_pdf_main.py, .\search_all_v3.py
```

### 3. 创建并激活虚拟环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

激活成功后，提示符通常会出现 `(.venv)`，并且以下命令应指向项目内的解释器：

```powershell
Get-Command python | Select-Object -ExpandProperty Source
```

预期路径以 `D:\math_model_kb\.venv\Scripts\python.exe` 结尾。若 PowerShell 阻止执行激活脚本，可只对当前进程放宽策略后重试：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以始终直接调用 `.\.venv\Scripts\python.exe`，无需修改执行策略。

### 4. 安装并核验依赖

先检查下载到的项目是否提供依赖文件：

```powershell
Get-ChildItem -File requirements*.txt, pyproject.toml, poetry.lock, Pipfile -ErrorAction SilentlyContinue
```

若存在经过项目验证的 `requirements.txt`：

```powershell
python -m pip install -r .\requirements.txt
```

若存在其他名称，例如 `requirements-windows.txt`，应使用项目文档指定的那一个：

```powershell
python -m pip install -r .\requirements-windows.txt
```

当前项目快照没有 `requirements.txt` 或 `pyproject.toml`，因此不能声称存在一条已经锁定、可完全复现的安装命令。源码的第三方 import 可通过以下命令复核：

```powershell
rg -n -g "*.py" "^(from|import) " .
```

如果没有安装 ripgrep（`rg`），可使用 PowerShell：

```powershell
Get-ChildItem -File -Recurse -Filter *.py |
    Select-String -Pattern '^(from|import) '
```

当前源码明确使用 OpenAI-compatible Python SDK、ChromaDB 和 Sentence Transformers，对应 import 分别为 `openai`、`chromadb`、`sentence_transformers`。PDF 入口还要求系统中能找到 `mineru` 命令。MinerU 的发行包名和安装方式应以你采用的 MinerU 版本的官方文档为准，本项目当前没有记录可验证的安装来源，因此这里不编造包名、版本或下载链接。

正确生成依赖清单的做法是：在新的 `.venv` 中，根据 import 和各依赖的官方安装说明逐项安装，运行下文的自检与一次完整评审；确认无误后再记录该环境：

```powershell
python -m pip check
python -m pip freeze | Set-Content -Encoding utf8 .\requirements.txt
```

提交 `requirements.txt` 前应人工删除与项目无关的包，并在另一个全新虚拟环境中执行 `python -m pip install -r requirements.txt` 复验。不要直接复制旧机器的整个 `.venv`，也不要仅凭 import 名称猜测 PyPI 包名或版本。

### 5. 安全配置 DeepSeek API Key

历史 PyCharm 截图曾显示 DeepSeek API Key，该 Key 应视为已暴露。请先在 DeepSeek 控制台撤销旧 Key并生成新 Key。不要把 Key 写进 Python 源码、README、命令脚本、`.env`、IDE 配置、Git 提交或截图。

仅在当前 PowerShell 会话中设置（关闭窗口后失效）：

```powershell
$secureKey = Read-Host "请输入新的 DEEPSEEK_API_KEY" -AsSecureString
$keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
    Remove-Variable secureKey, keyPtr -ErrorAction SilentlyContinue
}
```

这种写法不会把 Key 明文留在 PowerShell 命令历史中，但运行中的 Python 进程仍会通过环境变量取得明文，这是 API 客户端工作的必要条件。只检查变量是否存在，不要输出它的值：

```powershell
if ($env:DEEPSEEK_API_KEY) { "DEEPSEEK_API_KEY 已设置" } else { "DEEPSEEK_API_KEY 未设置" }
```

如需保存为当前 Windows 用户的持久环境变量，可在输入新 Key 后执行下列命令；新值只对之后启动的 PowerShell 和应用生效：

```powershell
$secureKey = Read-Host "请输入新的 DEEPSEEK_API_KEY" -AsSecureString
$keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr)
    [Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", $plainKey, "User")
} finally {
    if ($plainKey) { $plainKey = $null }
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
    Remove-Variable secureKey, keyPtr, plainKey -ErrorAction SilentlyContinue
}
```

设置后关闭并重新打开 PowerShell，再用上一段“是否存在”检查确认。若不再使用，可删除用户级变量：

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", $null, "User")
Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
```

可选变量包括诊断、报告、rubric、issue policy 与 parser artifact 等模型名称覆盖项，具体名称可在 `review_agent_main.py` 中搜索 `DEEPSEEK_` 查看。`.gitignore` 已排除 `.env*`（保留可选的 `.env.example`）和 `.idea/`，但忽略规则不能替代密钥轮换。

### 6. 准备本地模型、数据库与语料

完整 RAG 运行需要以下本地资源，它们体积较大或含论文资料，默认不纳入 Git：

```text
database/chroma/             # Chroma 向量数据库
models/bge-base-zh-v1.5/     # 本地 embedding 模型
CUMCM_20xx_?/                # 本地优秀论文语料
parsed_all/                  # MinerU 解析产物
processed_all/               # 结构化处理中间产物
```

这些资源不会随 GitHub 源码仓库自动提供。当前项目没有记录可公开验证的模型、Chroma 数据库或论文语料下载地址，因此使用者必须自行合法准备，不能根据本 README 猜测来源或使用虚构链接。已有本地资源时，按上面的目录结构放置：模型目录中应包含 Sentence Transformers 可加载的完整模型文件，`database/chroma/` 应是与 `search_all_v3.py` 中 collection 名称匹配的持久化数据库。

原始论文语料用于重建知识库，而日常评审实际读取 `database/chroma/` 与 `models/bge-base-zh-v1.5/`。如需自行重建，应依次审查并运行项目中的解析、结构化、切块和建库脚本；这些脚本同样使用本地绝对路径，且处理前必须确认论文资料的使用许可。

### 7. 入口与资源自检

确认正式入口版本：

```powershell
python -c "import review_agent_main as m; print(m.V3_LAYER_VERSION, m.JSON_RUNTIME_PATCH_VERSION)"
```

预期输出：

```text
3.0.6 1.0
```

检查必要资源和 MinerU 命令：

```powershell
$required = @(
    ".\review_agent_main.py",
    ".\review_pdf_main.py",
    ".\search_all_v3.py",
    ".\rubric_v1.json",
    ".\experience_library_v1_frozen.jsonl",
    ".\database\chroma",
    ".\models\bge-base-zh-v1.5"
)
$required | ForEach-Object {
    if (Test-Path -LiteralPath $_) { "[OK] $_" } else { "[MISSING] $_" }
}
Get-Command mineru -ErrorAction SilentlyContinue
python -m pip check
```

`Get-Command mineru` 没有输出表示 PDF 转换工具尚不可用；此时仍可使用已经准备好的 `.md`、`.markdown` 或 `.txt` 输入。

### 8. 运行评审

完整 PDF 示例（路径含空格时必须加引号）：

```powershell
Set-Location "D:\math_model_kb"
.\.venv\Scripts\Activate.ps1
python .\review_pdf_main.py "D:\papers\待评审论文.pdf"
```

使用已有 Markdown：

```powershell
python .\review_pdf_main.py "D:\papers\待评审论文.md"
```

PDF 首次运行会调用 MinerU，并把解析结果写入 `review_pdf_cache\<PDF文件名>\`；同名解析结果存在时，包装入口会复用其中的 Markdown。评审流水线还会在工作目录中生成或复用 consolidation、fact audit、issue policy、parser guard、RAG 和最终报告缓存。V3 signature 或 Parser Guard 输出变化时，后续缓存会自动失效并重建。首次运行通常明显更慢，并会产生 DeepSeek API 调用；不要把缓存和报告提交到公开仓库，因为其中可能含论文内容。

### 9. 常见故障排查

**`ModuleNotFoundError` / `No module named ...`**

确认虚拟环境已激活，并用同一个解释器安装和运行：

```powershell
Get-Command python | Select-Object -ExpandProperty Source
python -m pip --version
python -m pip check
```

根据报错中的 import 回到依赖官方文档安装对应包；不要盲目安装同名 PyPI 包。

**提示未找到 `mineru`**

运行 `Get-Command mineru`。若未找到，请按所采用 MinerU 版本的官方说明安装到当前 `.venv`，或先将 PDF 合法转换成 Markdown 后直接传入 `.md` 文件。

**未激活 `.venv` 或激活脚本被阻止**

重新执行第 3 步，或直接运行：

```powershell
.\.venv\Scripts\python.exe .\review_pdf_main.py "D:\papers\待评审论文.pdf"
```

**提示没有 `DEEPSEEK_API_KEY`、401、鉴权失败或 Key 无效**

只检查变量是否存在，不要打印值；确认旧 Key 已撤销、新 Key 有效，并在持久化设置后重新打开 PowerShell。若 API 服务返回鉴权错误，应在 DeepSeek 控制台检查 Key 状态和账户权限。

**模型或 Chroma 数据库缺失**

确认 `models\bge-base-zh-v1.5\` 和 `database\chroma\` 存在且内容完整。仅创建空目录不能工作；模型必须可被 Sentence Transformers 加载，数据库必须包含项目使用的 collection。当前 README 不提供未经核实的下载链接，缺少资源时需联系项目维护者或自行依法重建。

**找不到输入文件或路径包含空格**

使用 `Test-Path -LiteralPath "完整路径"` 检查文件，并始终用双引号包住路径：

```powershell
Test-Path -LiteralPath "D:\papers\比赛论文 最终版.pdf"
python .\review_pdf_main.py "D:\papers\比赛论文 最终版.pdf"
```

**项目不在 `D:\math_model_kb`**

当前入口和知识库脚本含该绝对路径。把项目移动到预期位置，或先统一修改 `ROOT`、`V3_ROOT`、数据库与模型路径配置；否则即使当前目录正确，程序仍可能去错误位置读取资源或写缓存。

## V3.0.6 验证流程

```powershell
python -m py_compile .\review_agent_main.py .\review_agent_main_v3_0_6.py .\review_pdf_main.py .\search_all_v3.py
python -c "from pathlib import Path; assert Path('review_agent_main.py').read_bytes() == Path('review_agent_main_v3_0_6.py').read_bytes()"
python .\review_pdf_main.py .\test4.pdf
```

真实 test4 完成后重点检查：

- `04e3_parser_artifact_guard_v306.json`；
- `review_report_v3.md`；
- `06_report_fact_audit_v3.json`；
- `06_report_meta_v3.json`。

这些运行输出可能含用户论文内容，默认由 `.gitignore` 排除。确认上述验收项全部通过后，再讨论冻结 V3.0.6。

## GitHub 上传前检查

本目录当前尚不是 Git 仓库。初始化和首次提交前建议执行：

```powershell
git init
git status --short --ignored
git add --dry-run .
git diff --cached --stat
```

随后人工确认暂存列表中不包含 `.idea/`、`.env*`、API Key、测试 PDF、论文语料、`models/`、`database/chroma/`、解析缓存或评审报告。推荐再使用 Gitleaks 等工具扫描一次；只有确认历史与待提交内容均无密钥后再创建提交和远程仓库。

## 数据与许可提示

本项目包含或依赖本地论文、测试 PDF、模型权重及其派生数据。上传 GitHub 前请分别核验源码、论文语料、模型和经验库的版权或许可证。`.gitignore` 只防止默认提交，不能授予再分发权。
