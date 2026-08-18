# 数学建模论文评审 Agent：V3.0.6 上下文压缩

> 更新时间：2026-08-17  
> 用途：在新对话中上传本文件，即可继续当前开发与测试，不必重新解释历史过程。

## 1. 项目目标与运行环境

用户正在开发一个“数学建模竞赛论文指导型评审 Agent”，目标不是直接代写论文，而是：

- 对整篇数学建模论文进行章节评审和全文审查；
- 用跨章节证据纠正“某节没写 = 全文没写”的误判；
- 从历年优秀论文知识库检索同类写法；
- 给出 P0/P1/P2 修改优先级；
- 对最终报告执行事实越权审计；
- 对 PDF→Markdown 解析伪影保持保守，不把 OCR/公式/表格解析错误直接算作论文缺陷。

用户本地环境：

- 项目目录：`D:\math_model_kb`
- Python：项目 `.venv\Scripts\python.exe`，Python 3.12
- 主入口：`review_agent_main.py`
- PDF 包装入口：`review_pdf_main.py`
- RAG 依赖：`search_all_v3.py`、`database/chroma`、`models/bge-base-zh-v1.5`
- API 环境变量：`DEEPSEEK_API_KEY`

主程序是单文件 bundled 版本，内部冻结并嵌入了历史 V2/V2.1/V2.3/V2.4/V2.5/V2.5.1 逻辑。

## 2. 当前最新交付物

当前最新程序：

- 文件：`review_agent_main_v3_0_6.py`
- `V3_LAYER_VERSION = "3.0.6"`
- `ISSUE_POLICY_LAYER_VERSION = "3.0.4"`
- `JSON_RUNTIME_PATCH_VERSION = "1.0"`
- 文件大小：339349 bytes
- SHA-256：`7b4f5692b979e733dcb1f30c516ff0b8bcd32c834e6b26cf27ff0d6c8d072bc3`

V3.0.6 尚未正式冻结；需用户在 Windows 上重新运行 test4，并回传新报告和 V3.0.6 Parser Guard JSON 后再决定冻结。

## 3. 当前评审流水线

1. 读取/复用 V2.5 consolidation：`04c_consolidated_issues_v25.json`
2. 跨章节事实审计：`04d_cross_section_fact_audit_v251.json`
3. 最终问题重叠审计：`04e_final_issues_v251.json`
4. V3 Issue Policy Guard：`04e2_issue_policy_v304.json`
5. V3.0.6 Parser Artifact Guard：`04e3_parser_artifact_guard_v306.json`
6. Rubric + Experience：`04f_rubric_experience_v3.json`
7. 原始 RAG + rerank：`05_retrieval_v3.json`
8. 生成报告并执行事实审计：
   - `review_report_v3.md`
   - `06_report_fact_audit_v3.json`
   - `06_report_meta_v3.json`

V3 signature 由版本、知识库签名、正式 issue 和 parser warnings 共同决定。Parser Guard 输出发生变化时，`04f`、`05` 和最终报告会自动失效并重建。

## 4. 已完成的核心改进

### 4.1 事实锁与跨章节核验

- 使用 `known_facts` 锁定已核验事实；
- 使用 `removed_claims` 和 `forbidden_phrases` 防止已被证据反驳的说法在后续 RAG 或报告中复活；
- 最终报告生成后再做 factual-claim audit；若发现越权陈述，自动修复一次并复审。

### 4.2 Issue Policy Guard（V3.0.4，继续复用）

- 对问题执行 `keep / correct / drop` 类策略审计；
- 收窄证据不足或措辞过重的问题；
- 保护带索引的符号族，避免把不同下标变量错误地当成同一个变量。

对应缓存仍为 `04e2_issue_policy_v304.json`，V3.0.6 不重新调用这一层。

### 4.3 Parser Artifact Guard（V3.0.5 → V3.0.6）

V3.0.5 已引入“原 PDF 待核验项”机制，但 test4 暴露出泛化不足：模型虽然标记了某些解析敏感字符，正式问题仍可能继续保留同一公式或表格指控。

V3.0.6 增加确定性兜底规则：

- **纯公式解析问题整体隔离**：公式重复项、缺项、异常下标、结构错位、乱码等，只要核心指控完全来自解析后的公式，必须 `quarantine`；
- **依赖表格结构的问题整体隔离**：若结论依赖解析后的表格行列、单元格、重复行列、起终点或类型划分，未回看原 PDF 前不得作为正式缺陷；
- **混合问题拆分**：若一个 issue 同时包含解析敏感公式问题和独立可靠问题，删除公式子问题，仅保留独立问题；
- **章节子句级清洗**：同一句中引用多个章节时，在章节编号前进一步切分，避免保留问题继承被隔离子问题的数字和证据；
- **缩写锚点修复**：Python 的 Unicode `\b` 无法正确识别中文旁边的 `HHI`，已改为 ASCII 大写字母边界判断。

新增/修改的关键函数：

- `_formula_problem_parts`
- `_is_formula_artifact_part`
- `_formula_refs_from_claims`
- `_is_formula_artifact_only_issue`
- `_is_table_structure_parser_issue`
- `_split_text_units`
- `_strip_formula_artifact_component`
- `guard_parser_artifacts`

### 4.4 DeepSeek JSON 运行时热修复

test4 曾在最终 factual audit 阶段失败，原因有两个：

1. DeepSeek JSON mode 要求提示词中显式出现英文 `json`；
2. 模型返回的 TeX（如 `\delta`、`\vphantom`）可能带原始反斜杠，导致 `Invalid \escape`，部分合法 JSON 单字母转义还可能把 TeX 静默变成制表符等字符。

已加入：

- `_escape_invalid_json_backslashes`
- `_extract_json_object_hotfix`
- `reliable_call_json_hotfix`

并 monkey-patch：

- `v21.reliable_call_json`
- `v21.base.call_json`

兼容策略：前几次使用 JSON mode；最后一次可退回普通模式后自行提取 JSON object；显式要求合法 JSON；修复 TeX 反斜杠，同时保留标准 JSON 转义、引号、Unicode 和换行。

## 5. test4 的问题与 V3.0.6 预期结果

用户提供的 V3.0.5 结果：

- `review_report_v3(10).md`
- `04e3_parser_artifact_guard_v305.json`

发现的问题：

| Issue | V3.0.5 问题 | V3.0.6 预期 |
| --- | --- | --- |
| I01 | 正式 P0 仍批评公式(33)重复项/错误下标，同时第五部分又说具体字符待核验 | 整个公式(33)问题进入第五部分，不再计入 P0/P1/P2 |
| I02 | 表12弧段重复、起点错误、类型重叠依赖解析后的表格单元格，却仍作为 P0 | 整体进入第五部分，等待查看原 PDF 表12 |
| I03 | “公式(32)乱码”和“HHI 定义不清”混在一个正式问题中 | 正式问题只保留“HHI定义表述不清”；公式(32)单独进入第五部分 |
| I04 | 模型变量定义不完整或不清晰 | 保留为正式 P1 |
| I05 | 模型构建依据不足 | 保留为正式 P1 |

I03 清洗后必须满足：

- `problem == "HHI定义表述不清"`
- 正式 issue 的 problem/reason/evidence/suggestion 中不再出现：`公式(32)`、`乱码`、`2900`、`5220`
- `source_sections` 只保留：
  - `4.5.1 目标函数`
  - `4.6.2 多维度改善效果`

## 6. 已完成的验证

### 6.1 语法验证

```text
python -m py_compile review_agent_main_v3_0_6.py
```

已通过。

### 6.2 真实 test4 守卫结果重放

使用用户上传的 `04e3_parser_artifact_guard_v305.json` 作为模型返回 fixture，调用 V3.0.6 的确定性守卫逻辑，断言结果：

```text
FORMAL   = [I03, I04, I05]
WARNINGS = [I01, I02, I03]
ALL_REPLAY_ASSERTIONS_PASSED
```

具体结果：

- I01 → QUARANTINE
- I02 → QUARANTINE
- I03 → CORRECT，仅保留 HHI
- I04 → CORRECT，正式 P1
- I05 → CORRECT，正式 P1

### 6.3 通用边界测试

已验证：

- 纯公式重复项/异常下标会被识别为公式伪影问题；
- “公式问题 + HHI 问题”不会被误判成纯公式问题；
- 表格弧段/单元格结构冲突会被隔离；
- 普通“表格结果缺少误差分析”不会被错误隔离；
- 混合问题只保留带 HHI 锚点的章节证据；
- 包在说明文字中的 JSON object 可以提取；
- 原始 TeX `\delta`、`\vphantom` 可以安全解析；
- Unicode 中文保持正常。

结果：

```text
ALL_GENERALIZED_EDGE_TESTS_PASSED
```

## 7. 用户本地替换与运行命令

将新文件放到 `D:\math_model_kb` 后，在 PowerShell 中运行：

```powershell
cd D:\math_model_kb
Copy-Item .\review_agent_main_v3_0_6.py .\review_agent_main.py -Force
python -c "import review_agent_main as m; print(m.V3_LAYER_VERSION, m.JSON_RUNTIME_PATCH_VERSION)"
python .\review_pdf_main.py .\test4.pdf
```

版本输出应为：

```text
3.0.6 1.0
```

缓存行为：

- 复用 `04e2_issue_policy_v304.json`；
- 新建 `04e3_parser_artifact_guard_v306.json`；
- 因 V3 signature 改变，重新生成 `04f_rubric_experience_v3.json`、`05_retrieval_v3.json`、`review_report_v3.md`、报告审计与元数据。

## 8. 下一步必须做的事

用户运行 V3.0.6 后，需要回传：

1. 新的 `review_report_v3.md`
2. 新的 `04e3_parser_artifact_guard_v306.json`

接手者应检查：

- I01、I02 是否只出现在“原 PDF 待核验项”；
- I03 正式问题是否只讨论 HHI；
- I03 是否已清除公式(32)、2900、5220 和无关章节；
- I04、I05 是否仍为正式 P1；
- 报告是否出现新的事实复活、优先级不一致或重复问题；
- `06_report_fact_audit_v3.json` 是否通过。

若全部通过，再讨论冻结 V3.0.6；不要在用户实际运行前宣称版本已冻结。

## 9. 重要安全事项

用户此前上传的 PyCharm 运行配置截图中显示了 DeepSeek API Key。该 Key 应视为已暴露：

- 立即在 DeepSeek 控制台撤销旧 Key；
- 生成新 Key；
- 更新 PyCharm 的 `DEEPSEEK_API_KEY` 环境变量；
- 后续截图应遮住密钥，不要把密钥写进代码或 Markdown。

## 10. 给新对话的最短续接提示

可在新对话上传本文件、`review_agent_main_v3_0_6.py`、新报告和 `04e3_parser_artifact_guard_v306.json`，并发送：

> 继续这个数学建模论文评审 Agent 项目。请按上下文文件审查我刚运行的 V3.0.6 输出，判断是否可以冻结；若不能，直接定位根因并修改程序。

