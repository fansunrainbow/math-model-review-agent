"""
review_agent_main.py

Standalone bundled entry point for the Whole-Paper Math Modeling Review Agent.

This file bundles the frozen V2.5.1 review-agent implementation and its
historical internal dependencies into one file. It still uses search_all_v3.py
as the reusable local-RAG search module.

Run:
    python review_agent_main.py "paper.md"

Requirements:
    - DEEPSEEK_API_KEY environment variable
    - D:/math_model_kb/database/chroma
    - D:/math_model_kb/models/bge-base-zh-v1.5
    - search_all_v3.py in D:/math_model_kb
"""

import sys
import types


def _load_embedded_module(name: str, source: str):
    """Create an in-memory module so legacy imports resolve without files."""
    module = types.ModuleType(name)
    module.__file__ = f"<embedded:{name}>"
    module.__package__ = ""
    sys.modules[name] = module
    code = compile(source, module.__file__, "exec")
    exec(code, module.__dict__)
    return module


# ===== Embedded module: review_agent_deepseek_v2 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2 = 'import json\nimport os\nimport re\nimport sys\nfrom collections import defaultdict\nfrom datetime import datetime\nfrom pathlib import Path\n\nimport chromadb\nfrom openai import OpenAI\nfrom sentence_transformers import SentenceTransformer\n\nfrom search_all_v3 import (\n    COLLECTION_NAME,\n    DB_PATH,\n    MODEL_NAME,\n    extract_body,\n    query_candidates,\n    rerank_results,\n)\n\n\n# ============================================================\n# 基础配置\n# ============================================================\n\nROOT = Path("D:/math_model_kb")\nOUTPUT_ROOT = ROOT / "reviews_v2"\n\nDEEPSEEK_BASE_URL = "https://api.deepseek.com"\n\nDIAG_MODEL = os.getenv(\n    "DEEPSEEK_DIAG_MODEL",\n    "deepseek-v4-flash",\n)\n\nREPORT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_MODEL",\n    "deepseek-v4-pro",\n)\n\nMAX_LOCAL_ISSUES_PER_SECTION = 3\nMAX_GLOBAL_ISSUES = 5\nMAX_FINAL_ISSUES = 10\nREFERENCES_PER_ISSUE = 3\n\n# 单次发送给诊断模型的正文最大字符数。\n# 超长章节会分片评审，最后统一合并。\nSECTION_PART_CHARS = 9000\n\n# 太短的章节不单独做局部评审，但仍会进入全文目录。\nMIN_REVIEW_SECTION_CHARS = 80\n\n\nSECTION_TYPES = {\n    "abstract",\n    "background",\n    "problem_statement",\n    "analysis",\n    "assumption",\n    "notation",\n    "preprocessing",\n    "model_idea",\n    "model_selection",\n    "model_building",\n    "model_solving",\n    "model_detail",\n    "result",\n    "validation",\n    "error_analysis",\n    "sensitivity",\n    "evaluation",\n    "strength",\n    "weakness",\n    "generalization",\n}\n\n\nFALLBACK_SECTION_TYPES = {\n    "model_selection": ["model_idea"],\n    "model_building": ["model_detail"],\n    "model_solving": ["model_detail"],\n    "validation": ["sensitivity", "error_analysis"],\n    "error_analysis": ["validation"],\n    "sensitivity": ["validation"],\n    "evaluation": ["strength", "weakness"],\n    "result": ["validation"],\n    "analysis": ["model_idea"],\n}\n\n\n# ============================================================\n# DeepSeek Prompts\n# ============================================================\n\nLOCAL_REVIEW_SYSTEM = r"""\n你是一名全国大学生数学建模竞赛论文评审助手。\n\n现在你评审的是“整篇论文中的某一个章节/章节片段”，不是整篇论文。\n\n最重要的规则：\n1. 你会同时看到整篇论文目录，因此绝对不要因为当前章节中没有出现某个内容，就断言整篇论文缺少该内容。\n   例如当前是“问题一模型建立”，即使这里没有灵敏度分析，只要目录中存在“灵敏度分析”，就不能批评“全文缺少灵敏度分析”。\n2. 局部评审只判断当前章节自身的问题，例如：\n   - 推导跳步；\n   - 模型选择没有依据；\n   - 数据处理没有解释；\n   - 变量定义不清；\n   - 目标函数或约束条件不完整；\n   - 求解过程不可复现；\n   - 结果只报数值没有解释；\n   - 本节内部论证不足。\n3. 不要为了凑数量挑标点、措辞、排版等小问题。\n4. 最多给出 3 个高价值问题；如果本节本身写得合理，可以少于 3 个，甚至 0 个。\n5. evidence_from_draft 必须来自当前章节真实内容，不得虚构。\n6. retrieval_query 应适合去“优秀数学建模论文知识库”检索同类写法。\n7. retrieval_section_type 必须从给定类型中选择一个。\n\n允许类型：\nabstract, background, problem_statement, analysis, assumption, notation,\npreprocessing, model_idea, model_selection, model_building, model_solving,\nmodel_detail, result, validation, error_analysis, sensitivity, evaluation,\nstrength, weakness, generalization\n\n只输出合法 JSON，不要输出 Markdown 或解释文字。\n\nJSON：\n{\n  "section_assessment": "对本章节的简短评价",\n  "issues": [\n    {\n      "problem": "问题名称",\n      "importance": "high|medium|low",\n      "reason": "为什么这是问题",\n      "evidence_from_draft": "当前章节中的原文或准确概括",\n      "retrieval_section_type": "model_building",\n      "retrieval_query": "用于搜索优秀论文案例的中文语句",\n      "suggested_direction": "应该朝什么方向修改，不直接代写正文"\n    }\n  ]\n}\n"""\n\n\nGLOBAL_REVIEW_SYSTEM = r"""\n你是一名全国大学生数学建模竞赛论文评审助手。\n\n现在进行的是“全文结构层面的检查”。\n\n你会看到：\n1. 整篇论文目录；\n2. 每个章节的类型、长度和少量内容预览。\n\n你的任务只检查必须从全文角度才能判断的问题，例如：\n- 摘要是否缺失或明显不完整；\n- 题目各问是否有遗漏；\n- 模型建立、求解、结果之间是否结构脱节；\n- 全文是否真的缺少模型检验、误差分析、灵敏度分析等关键环节；\n- 结论与前文是否缺少对应关系；\n- 各章节之间是否明显重复或逻辑顺序混乱。\n\n规则：\n1. 只有当目录和全文结构证据足够时，才能判断“全文缺少某部分”。\n2. 不要重复局部章节中容易发现的问题。\n3. 最多给出 5 个真正重要的全文问题。\n4. 不要为了凑数量制造问题。\n5. retrieval_section_type 用于后续从优秀论文知识库寻找同类参考。\n6. 只输出合法 JSON。\n\nJSON：\n{\n  "overall_structure_assessment": "全文结构总体评价",\n  "issues": [\n    {\n      "problem": "全文结构问题",\n      "importance": "high|medium|low",\n      "reason": "为什么重要",\n      "evidence_from_draft": "目录/章节结构中的证据",\n      "retrieval_section_type": "validation",\n      "retrieval_query": "用于搜索优秀论文案例的中文语句",\n      "suggested_direction": "修改方向"\n    }\n  ]\n}\n"""\n\n\nMERGE_SYSTEM = r"""\n你是一名数学建模竞赛论文总评助手。\n\n下面已经有若干“局部章节问题”和“全文结构问题”。\n\n你的任务：\n1. 合并语义重复的问题；\n2. 保留更具体、更有证据的问题；\n3. 按对竞赛论文质量的影响排序；\n4. 最终最多保留 10 个高价值问题；\n5. 不要新增原始评审中不存在的问题；\n6. 不要把两个本质不同的问题硬合并；\n7. 保留每个问题对应的来源章节和 evidence；\n8. retrieval_section_type 必须保留为合法类型；\n9. 只输出合法 JSON。\n\nJSON：\n{\n  "overall_assessment": "基于全部初步评审得到的总体判断",\n  "issues": [\n    {\n      "problem": "合并后的问题名称",\n      "importance": "high|medium|low",\n      "reason": "问题原因",\n      "evidence_from_draft": "草稿证据",\n      "source_sections": ["章节标题1", "章节标题2"],\n      "retrieval_section_type": "model_selection",\n      "retrieval_query": "用于优秀论文知识库检索的语句",\n      "suggested_direction": "修改方向"\n    }\n  ]\n}\n"""\n\n\nFINAL_REPORT_SYSTEM = r"""\n你是一名数学建模竞赛论文指导型评审助手。\n\n你会收到：\n1. 整篇论文目录；\n2. 已合并、排序的问题；\n3. 针对每个问题从历年优秀论文知识库检索出的真实案例。\n\n你的目标是指导作者自己修改，而不是直接替作者重写论文。\n\n最终报告必须包含：\n\n# 一、总体评价\n指出当前论文最主要的优点、短板，以及整体完成度。\n\n# 二、最优先修改的问题\n按重要性排序逐项写。每个问题必须包括：\n- 问题是什么；\n- 为什么会影响论文质量/比赛评分；\n- 草稿中的具体证据；\n- 优秀论文案例如何处理类似问题；\n- 从案例中抽象出的通用方法；\n- 作者应该按什么步骤修改。\n\n引用案例时必须写：\n- paper_id\n- section_title\n\n注意：\n- 只能依据提供的优秀论文检索内容；\n- 不要把优秀论文的具体模型机械套到当前论文；\n- 如果案例只提供结构或写法参考，要明确说明；\n- 不要声称知识库案例证明了草稿中没有提供的事实。\n\n# 三、分模块修改建议\n结合论文目录，按主要模块说明：\n摘要 / 问题分析 / 数据处理 / 模型建立 / 模型求解 / 结果分析 / 模型检验 / 模型评价\n哪些模块需要重点修改，哪些已经相对完整。\n\n# 四、修改顺序\n给出：\nP0：必须优先改\nP1：重点改\nP2：有时间再优化\n\n不要直接生成整篇替换版论文。\n"""\n\n\n# ============================================================\n# 通用 DeepSeek JSON 调用\n# ============================================================\n\ndef require_api_key():\n    key = os.getenv("DEEPSEEK_API_KEY")\n    if not key:\n        print(\n            "没有检测到 DEEPSEEK_API_KEY。\\n"\n            "PowerShell：\\n"\n            \'$env:DEEPSEEK_API_KEY="你的Key"\'\n        )\n        sys.exit(1)\n    return key\n\n\ndef make_client(api_key):\n    return OpenAI(\n        api_key=api_key,\n        base_url=DEEPSEEK_BASE_URL,\n    )\n\n\ndef strip_json_fence(text):\n    text = (text or "").strip()\n\n    if text.startswith("```"):\n        text = re.sub(\n            r"^```(?:json)?\\s*",\n            "",\n            text,\n            flags=re.I,\n        )\n        text = re.sub(\n            r"\\s*```$",\n            "",\n            text,\n        )\n\n    return text.strip()\n\n\ndef call_json(\n    client,\n    model,\n    system_prompt,\n    user_prompt,\n    max_tokens=5000,\n    retries=3,\n):\n    last_error = None\n    prompt = user_prompt\n\n    for attempt in range(1, retries + 1):\n        try:\n            response = client.chat.completions.create(\n                model=model,\n                messages=[\n                    {\n                        "role": "system",\n                        "content": system_prompt,\n                    },\n                    {\n                        "role": "user",\n                        "content": prompt,\n                    },\n                ],\n                response_format={\n                    "type": "json_object"\n                },\n                temperature=0.2,\n                max_tokens=max_tokens,\n            )\n\n            content = (\n                response.choices[0]\n                .message.content\n            )\n\n            if not content or not content.strip():\n                raise ValueError(\n                    "模型返回空 content"\n                )\n\n            content = strip_json_fence(\n                content\n            )\n\n            return json.loads(content)\n\n        except Exception as e:\n            last_error = e\n            print(\n                f"JSON 调用第 {attempt} 次失败："\n                f"{repr(e)}"\n            )\n\n            prompt = (\n                user_prompt\n                + "\\n\\n上一次输出未通过 JSON 校验。"\n                  "这一次只输出一个完整、合法 JSON object，"\n                  "不要代码块，不要额外说明。"\n            )\n\n    raise RuntimeError(\n        f"连续 {retries} 次 JSON 调用失败："\n        f"{repr(last_error)}"\n    )\n\n\n# ============================================================\n# 论文读取与章节切分\n# ============================================================\n\ndef read_paper(path_str):\n    path = Path(path_str)\n\n    if not path.exists():\n        raise FileNotFoundError(\n            f"文件不存在：{path}"\n        )\n\n    if path.suffix.lower() not in {\n        ".md",\n        ".markdown",\n        ".txt",\n    }:\n        raise ValueError(\n            "V2 当前支持 .md / .markdown / .txt。\\n"\n            "PDF 请先用 MinerU 转 Markdown。"\n        )\n\n    text = path.read_text(\n        encoding="utf-8",\n        errors="ignore",\n    )\n\n    return path, text\n\n\ndef clean_heading_title(line):\n    line = line.strip()\n\n    line = re.sub(\n        r"^#{1,6}\\s*",\n        "",\n        line,\n    )\n\n    return line.strip()\n\n\ndef looks_like_heading(line):\n    s = line.strip()\n\n    if not s:\n        return False\n\n    if len(s) > 90:\n        return False\n\n    if s.startswith((\n        "$$",\n        "<table",\n        "![](",\n        "![",\n    )):\n        return False\n\n    # Markdown 标题\n    if re.match(\n        r"^#{1,6}\\s+\\S",\n        s,\n    ):\n        return True\n\n    # 中文大章节：一、问题分析\n    if re.match(\n        r"^[一二三四五六七八九十]+[、．.]\\s*\\S+",\n        s,\n    ):\n        return True\n\n    # 第X章 / 第X问 / 第X部分\n    if re.match(\n        r"^第[一二三四五六七八九十百\\d]+[章节问部分]\\s*[:：、.\\s]?\\s*\\S*",\n        s,\n    ):\n        return True\n\n    # 1 标题 / 1.1 标题 / 4.3.2 标题\n    if re.match(\n        r"^\\d+(?:\\.\\d+){0,4}\\s+[\\u4e00-\\u9fffA-Za-z]",\n        s,\n    ):\n        return True\n\n    # 1、标题 / 1. 标题\n    if re.match(\n        r"^\\d+[、．.]\\s*[\\u4e00-\\u9fffA-Za-z]",\n        s,\n    ):\n        return True\n\n    # 常见无编号一级标题\n    common = [\n        "摘要",\n        "关键词",\n        "问题重述",\n        "问题分析",\n        "模型假设",\n        "符号说明",\n        "数据预处理",\n        "模型建立",\n        "模型求解",\n        "模型检验",\n        "灵敏度分析",\n        "误差分析",\n        "模型评价",\n        "模型推广",\n        "参考文献",\n        "附录",\n    ]\n\n    if s in common:\n        return True\n\n    return False\n\n\ndef infer_section_type(title):\n    t = re.sub(\n        r"\\s+",\n        "",\n        title,\n    )\n\n    if "摘要" in t:\n        return "abstract"\n\n    if (\n        "背景" in t\n        or "问题背景" in t\n    ):\n        return "background"\n\n    if (\n        "问题重述" in t\n        or "题目重述" in t\n        or "问题描述" in t\n    ):\n        return "problem_statement"\n\n    if (\n        "问题分析" in t\n        or "题目分析" in t\n    ):\n        return "analysis"\n\n    if "假设" in t:\n        return "assumption"\n\n    if (\n        "符号" in t\n        or "变量说明" in t\n    ):\n        return "notation"\n\n    if (\n        "数据预处理" in t\n        or "数据处理" in t\n        or "数据清洗" in t\n    ):\n        return "preprocessing"\n\n    if (\n        "模型比较" in t\n        or "模型选择" in t\n        or "模型选取" in t\n    ):\n        return "model_selection"\n\n    if (\n        "建模思路" in t\n        or "求解思路" in t\n        or "模型思路" in t\n    ):\n        return "model_idea"\n\n    if (\n        "目标函数" in t\n        or "约束条件" in t\n        or "模型建立" in t\n        or "模型构建" in t\n        or "建立模型" in t\n    ):\n        return "model_building"\n\n    if (\n        "模型求解" in t\n        or "求解算法" in t\n        or "算法求解" in t\n        or "求解过程" in t\n    ):\n        return "model_solving"\n\n    if (\n        "误差分析" in t\n        or "误差检验" in t\n    ):\n        return "error_analysis"\n\n    if (\n        "灵敏度" in t\n        or "敏感性" in t\n    ):\n        return "sensitivity"\n\n    if (\n        "模型检验" in t\n        or "模型验证" in t\n        or "结果验证" in t\n        or "拟合检验" in t\n        or "鲁棒性检验" in t\n    ):\n        return "validation"\n\n    if (\n        "模型优点" in t\n        or "优点" in t\n        or "优势" in t\n    ):\n        return "strength"\n\n    if (\n        "模型缺点" in t\n        or "不足" in t\n        or "局限" in t\n    ):\n        return "weakness"\n\n    if (\n        "模型推广" in t\n        or "推广" in t\n    ):\n        return "generalization"\n\n    if (\n        "模型评价" in t\n        or "评价" in t\n    ):\n        return "evaluation"\n\n    if (\n        "结果分析" in t\n        or "结果讨论" in t\n        or "结果" in t\n        or "结论" in t\n    ):\n        return "result"\n\n    if "分析" in t:\n        return "analysis"\n\n    return "model_detail"\n\n\ndef split_sections(text):\n    lines = text.splitlines()\n\n    sections = []\n    current_title = "前置内容"\n    current_lines = []\n\n    def flush():\n        nonlocal current_title, current_lines\n\n        content = "\\n".join(\n            current_lines\n        ).strip()\n\n        if content or current_title != "前置内容":\n            sections.append({\n                "title": current_title,\n                "content": content,\n                "section_type":\n                    infer_section_type(\n                        current_title\n                    ),\n            })\n\n        current_lines = []\n\n    for line in lines:\n        if looks_like_heading(line):\n            flush()\n            current_title = (\n                clean_heading_title(line)\n            )\n        else:\n            current_lines.append(line)\n\n    flush()\n\n    # 删除完全空章节\n    cleaned = []\n\n    for i, sec in enumerate(\n        sections,\n        start=1,\n    ):\n        if (\n            not sec["content"].strip()\n            and sec["title"] == "前置内容"\n        ):\n            continue\n\n        sec = dict(sec)\n        sec["section_index"] = i\n        sec["char_count"] = len(\n            sec["content"]\n        )\n        cleaned.append(sec)\n\n    # 重新编号\n    for i, sec in enumerate(\n        cleaned,\n        start=1,\n    ):\n        sec["section_index"] = i\n\n    return cleaned\n\n\ndef build_outline(sections):\n    lines = []\n\n    for sec in sections:\n        lines.append(\n            f"{sec[\'section_index\']:02d}. "\n            f"[{sec[\'section_type\']}] "\n            f"{sec[\'title\']} "\n            f"({sec[\'char_count\']} chars)"\n        )\n\n    return "\\n".join(lines)\n\n\ndef make_section_preview(sec, max_chars=500):\n    content = re.sub(\n        r"\\s+",\n        " ",\n        sec["content"]\n    ).strip()\n\n    if len(content) > max_chars:\n        content = content[:max_chars] + "..."\n\n    return (\n        f"[{sec[\'section_type\']}] "\n        f"{sec[\'title\']}\\n"\n        f"{content}"\n    )\n\n\ndef split_long_section(content):\n    if len(content) <= SECTION_PART_CHARS:\n        return [content]\n\n    parts = []\n\n    start = 0\n\n    while start < len(content):\n        end = min(\n            start + SECTION_PART_CHARS,\n            len(content),\n        )\n\n        # 尽量在段落处截断\n        if end < len(content):\n            cut = content.rfind(\n                "\\n\\n",\n                start,\n                end,\n            )\n\n            if (\n                cut != -1\n                and cut > start + 3000\n            ):\n                end = cut\n\n        parts.append(\n            content[start:end].strip()\n        )\n\n        start = end\n\n    return [\n        x for x in parts\n        if x\n    ]\n\n\n# ============================================================\n# 局部章节评审\n# ============================================================\n\ndef normalize_issue(\n    issue,\n    default_section_title,\n):\n    if not isinstance(issue, dict):\n        return None\n\n    required = [\n        "problem",\n        "importance",\n        "reason",\n        "evidence_from_draft",\n        "retrieval_section_type",\n        "retrieval_query",\n        "suggested_direction",\n    ]\n\n    if any(\n        key not in issue\n        for key in required\n    ):\n        return None\n\n    if (\n        issue["importance"]\n        not in {\n            "high",\n            "medium",\n            "low",\n        }\n    ):\n        issue["importance"] = "medium"\n\n    if (\n        issue["retrieval_section_type"]\n        not in SECTION_TYPES\n    ):\n        issue["retrieval_section_type"] = (\n            "model_detail"\n        )\n\n    issue["source_sections"] = [\n        default_section_title\n    ]\n\n    return issue\n\n\ndef review_section_part(\n    client,\n    outline_text,\n    section,\n    part_text,\n    part_index,\n    part_total,\n):\n    user_prompt = (\n        "===== 整篇论文目录 =====\\n"\n        f"{outline_text}\\n\\n"\n        "===== 当前评审章节 =====\\n"\n        f"章节标题：{section[\'title\']}\\n"\n        f"初步章节类型：{section[\'section_type\']}\\n"\n        f"片段：{part_index}/{part_total}\\n\\n"\n        "===== 当前章节正文 =====\\n"\n        f"{part_text}\\n"\n        "===== 当前章节正文结束 ====="\n    )\n\n    data = call_json(\n        client=client,\n        model=DIAG_MODEL,\n        system_prompt=LOCAL_REVIEW_SYSTEM,\n        user_prompt=user_prompt,\n        max_tokens=4500,\n    )\n\n    raw_issues = data.get(\n        "issues",\n        [],\n    )\n\n    normalized = []\n\n    for issue in raw_issues[\n        :MAX_LOCAL_ISSUES_PER_SECTION\n    ]:\n        x = normalize_issue(\n            issue,\n            section["title"],\n        )\n\n        if x:\n            x["source_kind"] = "local"\n            x["source_section_index"] = (\n                section["section_index"]\n            )\n            x["source_part"] = (\n                f"{part_index}/{part_total}"\n            )\n            normalized.append(x)\n\n    return {\n        "section_assessment":\n            data.get(\n                "section_assessment",\n                "",\n            ),\n        "issues":\n            normalized,\n    }\n\n\ndef review_all_sections(\n    client,\n    sections,\n    outline_text,\n):\n    print(\n        "\\n[1/5] 正在逐章节评审..."\n    )\n\n    results = []\n    all_issues = []\n\n    reviewable = [\n        sec\n        for sec in sections\n        if sec["char_count"]\n        >= MIN_REVIEW_SECTION_CHARS\n    ]\n\n    print(\n        f"共识别 {len(sections)} 个章节，"\n        f"其中 {len(reviewable)} 个进入局部评审。"\n    )\n\n    for order, sec in enumerate(\n        reviewable,\n        start=1,\n    ):\n        parts = split_long_section(\n            sec["content"]\n        )\n\n        print(\n            f"\\n[{order}/{len(reviewable)}] "\n            f"{sec[\'title\']} "\n            f"({sec[\'section_type\']}, "\n            f"{sec[\'char_count\']} chars, "\n            f"{len(parts)} part)"\n        )\n\n        section_result = {\n            "section_index":\n                sec["section_index"],\n            "title":\n                sec["title"],\n            "section_type":\n                sec["section_type"],\n            "char_count":\n                sec["char_count"],\n            "parts": [],\n        }\n\n        for p_idx, part in enumerate(\n            parts,\n            start=1,\n        ):\n            part_result = (\n                review_section_part(\n                    client=client,\n                    outline_text=outline_text,\n                    section=sec,\n                    part_text=part,\n                    part_index=p_idx,\n                    part_total=len(parts),\n                )\n            )\n\n            section_result[\n                "parts"\n            ].append(part_result)\n\n            all_issues.extend(\n                part_result["issues"]\n            )\n\n            print(\n                f"  part {p_idx}/{len(parts)}："\n                f"{len(part_result[\'issues\'])} 个问题"\n            )\n\n        results.append(\n            section_result\n        )\n\n    return results, all_issues\n\n\n# ============================================================\n# 全文结构评审\n# ============================================================\n\ndef review_global_structure(\n    client,\n    sections,\n    outline_text,\n):\n    print(\n        "\\n[2/5] 正在做全文结构检查..."\n    )\n\n    previews = "\\n\\n".join(\n        make_section_preview(sec)\n        for sec in sections\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline_text}\\n\\n"\n        "===== 各章节内容预览 =====\\n"\n        f"{previews}\\n"\n    )\n\n    data = call_json(\n        client=client,\n        model=DIAG_MODEL,\n        system_prompt=GLOBAL_REVIEW_SYSTEM,\n        user_prompt=user_prompt,\n        max_tokens=4500,\n    )\n\n    issues = []\n\n    for issue in data.get(\n        "issues",\n        [],\n    )[:MAX_GLOBAL_ISSUES]:\n        x = normalize_issue(\n            issue,\n            "全文结构",\n        )\n\n        if x:\n            x["source_kind"] = "global"\n            issues.append(x)\n\n    print(\n        f"全文结构问题：{len(issues)} 个"\n    )\n\n    return {\n        "overall_structure_assessment":\n            data.get(\n                "overall_structure_assessment",\n                "",\n            ),\n        "issues":\n            issues,\n    }\n\n\n# ============================================================\n# 合并与去重\n# ============================================================\n\ndef pre_dedupe_issues(issues):\n    """\n    先做一层很轻量的去重，避免把大量几乎相同的问题全交给模型。\n    """\n    seen = set()\n    result = []\n\n    for issue in issues:\n        key = (\n            re.sub(\n                r"\\s+",\n                "",\n                issue.get("problem", "")\n            ).lower(),\n            issue.get(\n                "retrieval_section_type",\n                ""\n            ),\n        )\n\n        if key in seen:\n            continue\n\n        seen.add(key)\n        result.append(issue)\n\n    return result\n\n\ndef merge_issues(\n    client,\n    local_issues,\n    global_issues,\n):\n    print(\n        "\\n[3/5] 正在合并、去重并排序问题..."\n    )\n\n    raw = (\n        list(local_issues)\n        + list(global_issues)\n    )\n\n    raw = pre_dedupe_issues(raw)\n\n    if not raw:\n        return {\n            "overall_assessment":\n                "未识别到明显高价值问题。",\n            "issues": [],\n        }\n\n    # 防止极端情况下 prompt 过大\n    raw = raw[:60]\n\n    user_prompt = (\n        "下面是对同一篇数学建模论文得到的初步问题列表。\\n"\n        "请合并语义重复项，并保留最重要的问题。\\n\\n"\n        + json.dumps(\n            raw,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    data = call_json(\n        client=client,\n        model=DIAG_MODEL,\n        system_prompt=MERGE_SYSTEM,\n        user_prompt=user_prompt,\n        max_tokens=6000,\n    )\n\n    issues = []\n\n    for issue in data.get(\n        "issues",\n        [],\n    )[:MAX_FINAL_ISSUES]:\n        if not isinstance(\n            issue,\n            dict,\n        ):\n            continue\n\n        if (\n            issue.get(\n                "retrieval_section_type"\n            )\n            not in SECTION_TYPES\n        ):\n            issue[\n                "retrieval_section_type"\n            ] = "model_detail"\n\n        if (\n            issue.get("importance")\n            not in {\n                "high",\n                "medium",\n                "low",\n            }\n        ):\n            issue["importance"] = (\n                "medium"\n            )\n\n        issue.setdefault(\n            "source_sections",\n            [],\n        )\n\n        issues.append(issue)\n\n    print(\n        f"合并后保留：{len(issues)} 个高价值问题"\n    )\n\n    return {\n        "overall_assessment":\n            data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues":\n            issues,\n    }\n\n\n# ============================================================\n# 本地 RAG\n# ============================================================\n\ndef initialize_retrieval():\n    print(\n        "\\n正在连接本地优秀论文知识库..."\n    )\n\n    chroma_client = (\n        chromadb.PersistentClient(\n            path=str(DB_PATH)\n        )\n    )\n\n    collection = (\n        chroma_client.get_collection(\n            name=COLLECTION_NAME\n        )\n    )\n\n    print(\n        f"知识库记录数量："\n        f"{collection.count()}"\n    )\n\n    print(\n        "正在加载本地 BGE..."\n    )\n\n    embedding_model = (\n        SentenceTransformer(\n            MODEL_NAME,\n            device="cpu",\n        )\n    )\n\n    print(\n        "BGE 加载完成。"\n    )\n\n    return (\n        collection,\n        embedding_model,\n    )\n\n\ndef rows_to_examples(rows):\n    examples = []\n\n    for row in rows:\n        meta = row["metadata"]\n\n        examples.append({\n            "paper_id":\n                meta.get("paper_id"),\n            "problem_id":\n                meta.get("problem_id"),\n            "question_number":\n                meta.get(\n                    "question_number"\n                ),\n            "section_type":\n                meta.get(\n                    "section_type"\n                ),\n            "section_title":\n                meta.get(\n                    "section_title"\n                ),\n            "chunk_id":\n                row["chunk_id"],\n            "distance":\n                round(\n                    float(\n                        row["distance"]\n                    ),\n                    4,\n                ),\n            "content":\n                extract_body(\n                    row["document"]\n                )[:2400],\n        })\n\n    return examples\n\n\ndef retrieve_one_type(\n    issue,\n    section_type,\n    embedding_model,\n    collection,\n    top_k,\n):\n    rows, where = query_candidates(\n        model=embedding_model,\n        collection=collection,\n        query=issue["retrieval_query"],\n        section_types=[\n            section_type\n        ],\n        candidate_k=60,\n    )\n\n    selected, rejected = (\n        rerank_results(\n            rows=rows,\n            section_types=[\n                section_type\n            ],\n            top_k=top_k,\n        )\n    )\n\n    return selected, {\n        "where": where,\n        "candidate_count":\n            len(rows),\n        "rejected_count":\n            len(rejected),\n    }\n\n\ndef retrieve_examples_for_issue(\n    issue,\n    embedding_model,\n    collection,\n):\n    primary = issue[\n        "retrieval_section_type"\n    ]\n\n    selected, stats = retrieve_one_type(\n        issue=issue,\n        section_type=primary,\n        embedding_model=embedding_model,\n        collection=collection,\n        top_k=REFERENCES_PER_ISSUE,\n    )\n\n    used_papers = {\n        row["metadata"].get(\n            "paper_id"\n        )\n        for row in selected\n    }\n\n    fallback_used = []\n\n    for fallback in (\n        FALLBACK_SECTION_TYPES\n        .get(primary, [])\n    ):\n        if (\n            len(selected)\n            >= REFERENCES_PER_ISSUE\n        ):\n            break\n\n        extras, _ = retrieve_one_type(\n            issue=issue,\n            section_type=fallback,\n            embedding_model=embedding_model,\n            collection=collection,\n            top_k=5,\n        )\n\n        for row in extras:\n            paper_id = (\n                row["metadata"]\n                .get("paper_id")\n            )\n\n            if paper_id in used_papers:\n                continue\n\n            selected.append(row)\n            used_papers.add(\n                paper_id\n            )\n            fallback_used.append(\n                fallback\n            )\n\n            if (\n                len(selected)\n                >= REFERENCES_PER_ISSUE\n            ):\n                break\n\n    return {\n        "query":\n            issue["retrieval_query"],\n        "primary_section_type":\n            primary,\n        "fallback_used":\n            fallback_used,\n        "candidate_count":\n            stats["candidate_count"],\n        "rejected_count":\n            stats["rejected_count"],\n        "examples":\n            rows_to_examples(\n                selected[\n                    :REFERENCES_PER_ISSUE\n                ]\n            ),\n    }\n\n\ndef retrieve_for_all_issues(\n    merged_data,\n    embedding_model,\n    collection,\n):\n    print(\n        "\\n[4/5] 正在为最终问题检索优秀论文案例..."\n    )\n\n    enriched = []\n\n    issues = merged_data.get(\n        "issues",\n        [],\n    )\n\n    for i, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        print(\n            f"\\n[{i}/{len(issues)}] "\n            f"{issue[\'problem\']}"\n        )\n        print(\n            "类型：",\n            issue[\n                "retrieval_section_type"\n            ],\n        )\n        print(\n            "Query：",\n            issue["retrieval_query"],\n        )\n\n        retrieval = (\n            retrieve_examples_for_issue(\n                issue=issue,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        print(\n            "案例数：",\n            len(\n                retrieval[\n                    "examples"\n                ]\n            ),\n        )\n\n        x = dict(issue)\n        x["retrieval"] = retrieval\n        enriched.append(x)\n\n    return {\n        "overall_assessment":\n            merged_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues":\n            enriched,\n    }\n\n\n# ============================================================\n# 最终报告\n# ============================================================\n\ndef build_reference_text(\n    enriched_data,\n):\n    blocks = []\n\n    for i, issue in enumerate(\n        enriched_data.get(\n            "issues",\n            [],\n        ),\n        start=1,\n    ):\n        blocks.append(\n            f"## Issue {i}: "\n            f"{issue.get(\'problem\')}"\n        )\n\n        blocks.append(\n            f"importance: "\n            f"{issue.get(\'importance\')}"\n        )\n\n        blocks.append(\n            "source_sections: "\n            + ", ".join(\n                issue.get(\n                    "source_sections",\n                    []\n                )\n            )\n        )\n\n        blocks.append(\n            "reason: "\n            f"{issue.get(\'reason\')}"\n        )\n\n        blocks.append(\n            "evidence: "\n            f"{issue.get(\'evidence_from_draft\')}"\n        )\n\n        blocks.append(\n            "suggested_direction: "\n            f"{issue.get(\'suggested_direction\')}"\n        )\n\n        examples = (\n            issue.get(\n                "retrieval",\n                {}\n            ).get(\n                "examples",\n                []\n            )\n        )\n\n        if not examples:\n            blocks.append(\n                "RAG examples: none"\n            )\n            continue\n\n        for j, ex in enumerate(\n            examples,\n            start=1,\n        ):\n            blocks.append(\n                f"\\nExample {j}\\n"\n                f"paper_id: "\n                f"{ex[\'paper_id\']}\\n"\n                f"section_title: "\n                f"{ex[\'section_title\']}\\n"\n                f"section_type: "\n                f"{ex[\'section_type\']}\\n"\n                f"distance: "\n                f"{ex[\'distance\']}\\n"\n                f"content:\\n"\n                f"{ex[\'content\']}\\n"\n            )\n\n    return "\\n".join(blocks)\n\n\ndef generate_final_report(\n    client,\n    outline_text,\n    enriched_data,\n):\n    print(\n        f"\\n[5/5] 正在用 {REPORT_MODEL} "\n        "生成整篇论文评审报告..."\n    )\n\n    reference_text = (\n        build_reference_text(\n            enriched_data\n        )\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline_text}\\n\\n"\n        "===== 合并后的问题与优秀论文案例 =====\\n"\n        f"{reference_text}\\n"\n    )\n\n    response = (\n        client.chat.completions.create(\n            model=REPORT_MODEL,\n            messages=[\n                {\n                    "role": "system",\n                    "content":\n                        FINAL_REPORT_SYSTEM,\n                },\n                {\n                    "role": "user",\n                    "content":\n                        user_prompt,\n                },\n            ],\n            temperature=0.3,\n            max_tokens=9000,\n        )\n    )\n\n    content = (\n        response.choices[0]\n        .message.content\n    )\n\n    if not content or not content.strip():\n        raise RuntimeError(\n            "最终报告为空"\n        )\n\n    return content.strip()\n\n\n# ============================================================\n# 保存调试文件\n# ============================================================\n\ndef save_json(path, data):\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef save_outputs(\n    paper_path,\n    sections,\n    local_reviews,\n    global_review,\n    merged_data,\n    enriched_data,\n    final_report,\n):\n    timestamp = (\n        datetime.now()\n        .strftime(\n            "%Y%m%d_%H%M%S"\n        )\n    )\n\n    run_dir = (\n        OUTPUT_ROOT\n        / f"{paper_path.stem}_{timestamp}"\n    )\n\n    run_dir.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n\n    save_json(\n        run_dir / "01_outline.json",\n        {\n            "paper":\n                str(paper_path),\n            "sections":\n                sections,\n        },\n    )\n\n    save_json(\n        run_dir\n        / "02_section_reviews.json",\n        local_reviews,\n    )\n\n    save_json(\n        run_dir\n        / "03_global_review.json",\n        global_review,\n    )\n\n    save_json(\n        run_dir\n        / "04_merged_issues.json",\n        merged_data,\n    )\n\n    save_json(\n        run_dir\n        / "05_retrieval.json",\n        enriched_data,\n    )\n\n    report_path = (\n        run_dir\n        / "review_report.md"\n    )\n\n    report_path.write_text(\n        final_report,\n        encoding="utf-8",\n    )\n\n    return run_dir, report_path\n\n\n# ============================================================\n# 主程序\n# ============================================================\n\ndef main():\n    api_key = require_api_key()\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python review_agent_deepseek_v2.py "\n            "你的论文.md"\n        )\n        sys.exit(1)\n\n    paper_path, text = read_paper(\n        sys.argv[1]\n    )\n\n    print("=" * 72)\n    print("数学建模整篇论文评审 Agent V2")\n    print("=" * 72)\n    print("论文：", paper_path)\n    print("总字符数：", len(text))\n    print("诊断模型：", DIAG_MODEL)\n    print("最终报告模型：", REPORT_MODEL)\n\n    sections = split_sections(text)\n\n    if not sections:\n        raise RuntimeError(\n            "没有识别到任何章节。"\n        )\n\n    outline_text = build_outline(\n        sections\n    )\n\n    print()\n    print("=" * 72)\n    print("识别到的论文目录")\n    print("=" * 72)\n    print(outline_text)\n\n    client = make_client(\n        api_key\n    )\n\n    local_reviews, local_issues = (\n        review_all_sections(\n            client=client,\n            sections=sections,\n            outline_text=outline_text,\n        )\n    )\n\n    global_review = (\n        review_global_structure(\n            client=client,\n            sections=sections,\n            outline_text=outline_text,\n        )\n    )\n\n    merged_data = merge_issues(\n        client=client,\n        local_issues=local_issues,\n        global_issues=\n            global_review["issues"],\n    )\n\n    collection, embedding_model = (\n        initialize_retrieval()\n    )\n\n    enriched_data = (\n        retrieve_for_all_issues(\n            merged_data=merged_data,\n            embedding_model=\n                embedding_model,\n            collection=collection,\n        )\n    )\n\n    final_report = (\n        generate_final_report(\n            client=client,\n            outline_text=outline_text,\n            enriched_data=enriched_data,\n        )\n    )\n\n    run_dir, report_path = (\n        save_outputs(\n            paper_path=paper_path,\n            sections=sections,\n            local_reviews=\n                local_reviews,\n            global_review=\n                global_review,\n            merged_data=\n                merged_data,\n            enriched_data=\n                enriched_data,\n            final_report=\n                final_report,\n        )\n    )\n\n    print()\n    print("=" * 72)\n    print("最终评审报告")\n    print("=" * 72)\n    print()\n    print(final_report)\n\n    print()\n    print("=" * 72)\n    print("全部结果保存到：")\n    print(run_dir)\n    print("最终 Markdown：")\n    print(report_path)\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2 = _load_embedded_module('review_agent_deepseek_v2', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2)

# ===== Embedded module: review_agent_deepseek_v2_1 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2_1 = '"""\nreview_agent_deepseek_v2_1.py\n\n这是 review_agent_deepseek_v2.py 的增强/修复版。\n要求：\n1. D:/math_model_kb/review_agent_deepseek_v2.py 已存在\n2. D:/math_model_kb/search_all_v3.py 已存在\n\n修复点：\n- DeepSeek JSON Output 空 content：关闭 thinking + JSON模式失败后自动降级普通文本JSON\n- merge 阶段使用 V4-Pro，并改为分批合并，避免一次塞几十个 issue\n- 逐章节结果即时 checkpoint，后续失败可断点续跑\n- 跳过前置作者信息、参考文献、附录的局部评审\n- 改进章节类型识别\n"""\n\nimport hashlib\nimport json\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n\nimport review_agent_deepseek_v2 as base\n\n\nROOT = Path("D:/math_model_kb")\nWORK_ROOT = ROOT / "reviews_v2_work"\nFINAL_ROOT = ROOT / "reviews_v2"\n\nMERGE_MODEL = os.getenv(\n    "DEEPSEEK_MERGE_MODEL",\n    "deepseek-v4-pro",\n)\n\nMERGE_BATCH_SIZE = 15\n\n\n# ============================================================\n# 1. DeepSeek JSON：更稳健\n# ============================================================\n\ndef _extract_json_object(text: str):\n    """\n    既支持纯 JSON，也支持：\n    ```json\n    {...}\n    ```\n    或前后带少量说明的情况。\n    """\n    text = (text or "").strip()\n\n    if not text:\n        raise ValueError("模型返回空 content")\n\n    text = base.strip_json_fence(text)\n\n    try:\n        return json.loads(text)\n    except Exception:\n        pass\n\n    # 找第一个完整 JSON object，处理字符串中的花括号。\n    start = text.find("{")\n    if start < 0:\n        raise ValueError("返回内容中没有 JSON object")\n\n    depth = 0\n    in_string = False\n    escape = False\n\n    for i in range(start, len(text)):\n        ch = text[i]\n\n        if in_string:\n            if escape:\n                escape = False\n            elif ch == "\\\\":\n                escape = True\n            elif ch == \'"\':\n                in_string = False\n            continue\n\n        if ch == \'"\':\n            in_string = True\n        elif ch == "{":\n            depth += 1\n        elif ch == "}":\n            depth -= 1\n\n            if depth == 0:\n                candidate = text[start:i + 1]\n                return json.loads(candidate)\n\n    raise ValueError("没有找到完整 JSON object")\n\n\ndef reliable_call_json(\n    client,\n    model,\n    system_prompt,\n    user_prompt,\n    max_tokens=5000,\n    retries=3,\n):\n    """\n    策略：\n    1. 前两次：DeepSeek JSON Output + 显式关闭 thinking\n    2. 最后一次：不用 response_format，仍要求只输出 JSON，再自行解析\n       ——规避 DeepSeek 官方文档提到的 JSON Output 偶发空 content。\n    """\n    last_error = None\n\n    for attempt in range(1, retries + 1):\n        # 最后一次不用 JSON mode\n        use_json_mode = attempt < retries\n\n        repair_note = ""\n        if attempt > 1:\n            repair_note = (\n                "\\n\\n重要：上一次没有得到可解析结果。"\n                "本次必须直接输出一个完整 JSON object；"\n                "第一个字符应为 {，最后一个字符应为 }；"\n                "不要 Markdown 代码块，不要解释文字。"\n            )\n\n        messages = [\n            {\n                "role": "system",\n                "content": system_prompt,\n            },\n            {\n                "role": "user",\n                "content": user_prompt + repair_note,\n            },\n        ]\n\n        kwargs = dict(\n            model=model,\n            messages=messages,\n            temperature=0.2,\n            max_tokens=max_tokens,\n            # V4 默认 thinking enabled。\n            # 对纯结构化提取显式关闭，减少空 final content 的概率。\n            extra_body={\n                "thinking": {\n                    "type": "disabled"\n                }\n            },\n        )\n\n        if use_json_mode:\n            kwargs["response_format"] = {\n                "type": "json_object"\n            }\n\n        try:\n            response = client.chat.completions.create(\n                **kwargs\n            )\n\n            choice = response.choices[0]\n            content = choice.message.content\n\n            if not content or not content.strip():\n                finish_reason = getattr(\n                    choice,\n                    "finish_reason",\n                    None,\n                )\n                reasoning = getattr(\n                    choice.message,\n                    "reasoning_content",\n                    None,\n                )\n\n                raise ValueError(\n                    "模型返回空 content"\n                    f"，finish_reason={finish_reason}"\n                    f"，reasoning_content长度="\n                    f"{len(reasoning or \'\')}"\n                )\n\n            return _extract_json_object(\n                content\n            )\n\n        except Exception as e:\n            last_error = e\n\n            mode_name = (\n                "JSON mode"\n                if use_json_mode\n                else "text JSON fallback"\n            )\n\n            print(\n                f"JSON 调用第 {attempt} 次失败"\n                f"（{mode_name}）：{repr(e)}"\n            )\n\n    raise RuntimeError(\n        f"连续 {retries} 次 JSON 调用失败："\n        f"{repr(last_error)}"\n    )\n\n\n# 让原 V2 的局部评审、全文评审等全部自动使用新版本\nbase.call_json = reliable_call_json\n\n\n# ============================================================\n# 2. 更合理的章节类型\n# ============================================================\n\ndef infer_section_type_v21(title):\n    t = re.sub(r"\\s+", "", title)\n    t_no_num = re.sub(\n        r"^(?:第?[一二三四五六七八九十百\\d]+(?:[章节问、.．])?|\\d+(?:\\.\\d+)*[、.．]?)",\n        "",\n        t,\n    )\n\n    if title == "前置内容":\n        return "front_matter"\n\n    if "参考文献" in t:\n        return "reference"\n\n    if t.startswith("附录") or re.match(\n        r"^[A-ZＡ-Ｚ]\\.\\d+",\n        t,\n        flags=re.I,\n    ):\n        return "appendix"\n\n    # 只把真正的“摘要”标题判为 abstract，\n    # 避免 A.2 问题二结果摘要 被当成摘要。\n    if t_no_num in {"摘要", "摘要：", "摘要:"}:\n        return "abstract"\n\n    if "背景" in t:\n        return "background"\n\n    if any(x in t for x in [\n        "问题重述",\n        "题目重述",\n        "任务重述",\n        "任务描述",\n        "问题描述",\n    ]):\n        return "problem_statement"\n\n    if any(x in t for x in [\n        "问题分析",\n        "题目分析",\n        "任务分析",\n    ]):\n        return "analysis"\n\n    if "假设" in t:\n        return "assumption"\n\n    if any(x in t for x in [\n        "符号说明",\n        "变量说明",\n        "符号定义",\n    ]):\n        return "notation"\n\n    if any(x in t for x in [\n        "数据预处理",\n        "数据处理",\n        "数据清洗",\n        "异常值处理",\n    ]):\n        return "preprocessing"\n\n    if any(x in t for x in [\n        "模型比较",\n        "模型选择",\n        "模型选取",\n    ]):\n        return "model_selection"\n\n    if any(x in t for x in [\n        "建模思路",\n        "求解思路",\n        "模型思路",\n        "建模路线",\n        "技术路线",\n    ]):\n        return "model_idea"\n\n    if any(x in t for x in [\n        "误差分析",\n        "误差检验",\n    ]):\n        return "error_analysis"\n\n    if any(x in t for x in [\n        "灵敏度",\n        "敏感性",\n    ]):\n        return "sensitivity"\n\n    if any(x in t for x in [\n        "模型检验",\n        "模型验证",\n        "结果验证",\n        "拟合检验",\n        "鲁棒性检验",\n        "蒙特卡洛验证",\n        "验证层级",\n        "验证分析",\n    ]):\n        return "validation"\n\n    if any(x in t for x in [\n        "模型优点",\n        "优点",\n        "优势",\n    ]):\n        return "strength"\n\n    if any(x in t for x in [\n        "模型缺点",\n        "不足",\n        "局限",\n    ]):\n        return "weakness"\n\n    if any(x in t for x in [\n        "模型推广",\n        "推广应用",\n    ]):\n        return "generalization"\n\n    if "模型评价" in t:\n        return "evaluation"\n\n    # 求解要放在“建立”之前判断\n    if any(x in t for x in [\n        "模型求解",\n        "求解算法",\n        "算法求解",\n        "求解过程",\n        "参数校准",\n        "参数估计",\n        "样本平均近似",\n    ]):\n        return "model_solving"\n\n    if any(x in t for x in [\n        "目标函数",\n        "约束条件",\n        "模型建立",\n        "模型构建",\n        "建立模型",\n        "整数规划",\n        "动态规划模型",\n        "需求模型",\n        "优化模型",\n    ]):\n        return "model_building"\n\n    if any(x in t for x in [\n        "结果分析",\n        "结果讨论",\n        "求解结果",\n        "最优结果",\n        "结果与解释",\n        "结论",\n        "策略建议",\n        "收益提升",\n        "价格路径",\n        "结果摘要",\n    ]):\n        return "result"\n\n    if "分析" in t:\n        return "analysis"\n\n    return "model_detail"\n\n\nbase.infer_section_type = infer_section_type_v21\n\n\ndef split_sections_v21(text):\n    """\n    在原切分基础上增加三种内部区域：\n      front_matter：摘要之前的封面/队员信息\n      reference：参考文献\n      appendix：附录及其 A.x 子章节\n\n    这些区域保留在全文目录中，但不做逐节 AI 评审。\n    """\n    lines = text.splitlines()\n\n    raw = []\n    current_title = "前置内容"\n    current_lines = []\n\n    def flush():\n        nonlocal current_lines, current_title\n\n        content = "\\n".join(\n            current_lines\n        ).strip()\n\n        if content or current_title != "前置内容":\n            raw.append({\n                "title": current_title,\n                "content": content,\n            })\n\n        current_lines = []\n\n    for line in lines:\n        if base.looks_like_heading(line):\n            flush()\n            current_title = (\n                base.clean_heading_title(line)\n            )\n        else:\n            current_lines.append(line)\n\n    flush()\n\n    cleaned = []\n\n    body_started = False\n    mode = "front_matter"\n\n    for sec in raw:\n        title = sec["title"]\n        normal_type = infer_section_type_v21(\n            title\n        )\n\n        if normal_type == "abstract":\n            body_started = True\n            mode = "body"\n\n        if "参考文献" in re.sub(r"\\s+", "", title):\n            mode = "reference"\n\n        if (\n            re.sub(r"\\s+", "", title).startswith("附录")\n            or normal_type == "appendix"\n        ):\n            mode = "appendix"\n\n        if not body_started:\n            section_type = "front_matter"\n        elif mode == "reference":\n            section_type = "reference"\n        elif mode == "appendix":\n            section_type = "appendix"\n        else:\n            section_type = normal_type\n\n        content = sec["content"].strip()\n\n        # 前置空壳不保留\n        if (\n            not content\n            and title == "前置内容"\n        ):\n            continue\n\n        cleaned.append({\n            "title": title,\n            "content": content,\n            "section_type": section_type,\n            "char_count": len(content),\n        })\n\n    for i, sec in enumerate(\n        cleaned,\n        start=1,\n    ):\n        sec["section_index"] = i\n\n    return cleaned\n\n\nbase.split_sections = split_sections_v21\n\n\n# ============================================================\n# 3. 哪些章节值得逐节评审\n# ============================================================\n\nSKIP_LOCAL_TYPES = {\n    "front_matter",\n    "reference",\n    "appendix",\n}\n\n\ndef is_reviewable_section(sec):\n    if sec["section_type"] in SKIP_LOCAL_TYPES:\n        return False\n\n    if (\n        sec["char_count"]\n        < base.MIN_REVIEW_SECTION_CHARS\n    ):\n        return False\n\n    # 关键词一类不单独评审\n    if (\n        "关键词" in sec["title"]\n        and sec["char_count"] < 200\n    ):\n        return False\n\n    return True\n\n\n# ============================================================\n# 4. 断点续跑：逐章节评审即时保存\n# ============================================================\n\ndef _paper_hash(text):\n    return hashlib.sha256(\n        text.encode("utf-8", errors="ignore")\n    ).hexdigest()[:12]\n\n\ndef make_work_dir(\n    paper_path,\n    text,\n):\n    work_dir = (\n        WORK_ROOT\n        / f"{paper_path.stem}_{_paper_hash(text)}"\n    )\n    work_dir.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n    return work_dir\n\n\ndef save_json(path, data):\n    path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path, default=None):\n    if not path.exists():\n        return default\n\n    try:\n        return json.loads(\n            path.read_text(\n                encoding="utf-8"\n            )\n        )\n    except Exception:\n        return default\n\n\ndef review_all_sections_resume(\n    client,\n    sections,\n    outline_text,\n    checkpoint_path,\n):\n    print(\n        "\\n[1/5] 正在逐章节评审..."\n    )\n\n    reviewable = [\n        sec\n        for sec in sections\n        if is_reviewable_section(sec)\n    ]\n\n    checkpoint = load_json(\n        checkpoint_path,\n        default={\n            "completed": {},\n            "results": [],\n        },\n    )\n\n    completed = checkpoint.get(\n        "completed",\n        {},\n    )\n\n    result_by_key = {\n        x["checkpoint_key"]: x\n        for x in checkpoint.get(\n            "results",\n            [],\n        )\n        if "checkpoint_key" in x\n    }\n\n    print(\n        f"共识别 {len(sections)} 个章节，"\n        f"{len(reviewable)} 个进入局部评审；"\n        f"已缓存 {len(completed)} 个章节。"\n    )\n\n    for order, sec in enumerate(\n        reviewable,\n        start=1,\n    ):\n        key = (\n            f"{sec[\'section_index\']}|"\n            f"{sec[\'title\']}|"\n            f"{sec[\'char_count\']}"\n        )\n\n        if completed.get(key):\n            print(\n                f"\\n[{order}/{len(reviewable)}] "\n                f"{sec[\'title\']} —— 已有缓存，跳过"\n            )\n            continue\n\n        parts = base.split_long_section(\n            sec["content"]\n        )\n\n        print(\n            f"\\n[{order}/{len(reviewable)}] "\n            f"{sec[\'title\']} "\n            f"({sec[\'section_type\']}, "\n            f"{sec[\'char_count\']} chars, "\n            f"{len(parts)} part)"\n        )\n\n        section_result = {\n            "checkpoint_key": key,\n            "section_index":\n                sec["section_index"],\n            "title": sec["title"],\n            "section_type":\n                sec["section_type"],\n            "char_count":\n                sec["char_count"],\n            "parts": [],\n        }\n\n        for p_idx, part in enumerate(\n            parts,\n            start=1,\n        ):\n            part_result = (\n                base.review_section_part(\n                    client=client,\n                    outline_text=outline_text,\n                    section=sec,\n                    part_text=part,\n                    part_index=p_idx,\n                    part_total=len(parts),\n                )\n            )\n\n            section_result[\n                "parts"\n            ].append(\n                part_result\n            )\n\n            print(\n                f"  part {p_idx}/{len(parts)}："\n                f"{len(part_result[\'issues\'])} 个问题"\n            )\n\n        result_by_key[key] = (\n            section_result\n        )\n        completed[key] = True\n\n        # 每完成一个章节立即落盘\n        save_json(\n            checkpoint_path,\n            {\n                "completed": completed,\n                "results": list(\n                    result_by_key.values()\n                ),\n            },\n        )\n\n    results = list(\n        result_by_key.values()\n    )\n\n    # 按 section_index 排序\n    results.sort(\n        key=lambda x: x.get(\n            "section_index",\n            10**9,\n        )\n    )\n\n    all_issues = []\n\n    for section_result in results:\n        for part in section_result.get(\n            "parts",\n            [],\n        ):\n            all_issues.extend(\n                part.get(\n                    "issues",\n                    [],\n                )\n            )\n\n    return results, all_issues\n\n\n# ============================================================\n# 5. 全文预览：不把参考文献/附录正文塞进去\n# ============================================================\n\ndef review_global_structure_v21(\n    client,\n    sections,\n    outline_text,\n):\n    print(\n        "\\n[2/5] 正在做全文结构检查..."\n    )\n\n    preview_sections = [\n        sec for sec in sections\n        if sec["section_type"]\n        not in {\n            "front_matter",\n            "reference",\n            "appendix",\n        }\n    ]\n\n    previews = "\\n\\n".join(\n        base.make_section_preview(sec)\n        for sec in preview_sections\n    )\n\n    user_prompt = (\n        "===== 论文完整目录 =====\\n"\n        f"{outline_text}\\n\\n"\n        "注意：目录中的 front_matter/reference/appendix "\n        "只是用于确认论文结构，不应作为正文质量问题。\\n\\n"\n        "===== 正文各章节内容预览 =====\\n"\n        f"{previews}\\n"\n    )\n\n    data = reliable_call_json(\n        client=client,\n        model=base.DIAG_MODEL,\n        system_prompt=\n            base.GLOBAL_REVIEW_SYSTEM,\n        user_prompt=user_prompt,\n        max_tokens=4500,\n    )\n\n    issues = []\n\n    for issue in data.get(\n        "issues",\n        [],\n    )[:base.MAX_GLOBAL_ISSUES]:\n        x = base.normalize_issue(\n            issue,\n            "全文结构",\n        )\n\n        if x:\n            x["source_kind"] = (\n                "global"\n            )\n            issues.append(x)\n\n    print(\n        f"全文结构问题：{len(issues)} 个"\n    )\n\n    return {\n        "overall_structure_assessment":\n            data.get(\n                "overall_structure_assessment",\n                "",\n            ),\n        "issues": issues,\n    }\n\n\n# ============================================================\n# 6. 分批合并 issue\n# ============================================================\n\ndef _importance_rank(issue):\n    return {\n        "high": 0,\n        "medium": 1,\n        "low": 2,\n    }.get(\n        issue.get(\n            "importance",\n            "medium",\n        ),\n        1,\n    )\n\n\ndef _normalize_merged_data(data):\n    issues = []\n\n    for issue in data.get(\n        "issues",\n        [],\n    )[:base.MAX_FINAL_ISSUES]:\n        if not isinstance(\n            issue,\n            dict,\n        ):\n            continue\n\n        if (\n            issue.get(\n                "retrieval_section_type"\n            )\n            not in base.SECTION_TYPES\n        ):\n            issue[\n                "retrieval_section_type"\n            ] = "model_detail"\n\n        if (\n            issue.get("importance")\n            not in {\n                "high",\n                "medium",\n                "low",\n            }\n        ):\n            issue["importance"] = (\n                "medium"\n            )\n\n        issue.setdefault(\n            "source_sections",\n            [],\n        )\n        issue.setdefault(\n            "suggested_direction",\n            "",\n        )\n        issue.setdefault(\n            "evidence_from_draft",\n            "",\n        )\n        issue.setdefault(\n            "reason",\n            "",\n        )\n        issue.setdefault(\n            "retrieval_query",\n            issue.get(\n                "problem",\n                "",\n            ),\n        )\n\n        issues.append(issue)\n\n    return {\n        "overall_assessment":\n            data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues":\n            issues,\n    }\n\n\ndef _merge_one_batch(\n    client,\n    issues,\n    label,\n):\n    user_prompt = (\n        f"这是同一篇论文的{label}。\\n"\n        "请合并语义重复的问题，保留具体证据，"\n        "并优先保留 high / medium 的高价值问题。\\n"\n        "只根据给定问题合并，不要新增问题。\\n\\n"\n        + json.dumps(\n            issues,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    data = reliable_call_json(\n        client=client,\n        # 合并是关键决策，少量调用使用 Pro\n        model=MERGE_MODEL,\n        system_prompt=\n            base.MERGE_SYSTEM,\n        user_prompt=user_prompt,\n        max_tokens=6500,\n        retries=3,\n    )\n\n    return _normalize_merged_data(\n        data\n    )\n\n\ndef merge_issues_v21(\n    client,\n    local_issues,\n    global_issues,\n):\n    print(\n        "\\n[3/5] 正在分批合并、去重并排序问题..."\n    )\n\n    raw = (\n        list(local_issues)\n        + list(global_issues)\n    )\n\n    raw = base.pre_dedupe_issues(\n        raw\n    )\n\n    if not raw:\n        return {\n            "overall_assessment":\n                "未识别到明显高价值问题。",\n            "issues": [],\n        }\n\n    raw.sort(\n        key=_importance_rank\n    )\n\n    # 不再一次把 60 条全塞进去。\n    batches = [\n        raw[i:i + MERGE_BATCH_SIZE]\n        for i in range(\n            0,\n            len(raw),\n            MERGE_BATCH_SIZE,\n        )\n    ]\n\n    print(\n        f"初步问题 {len(raw)} 个，"\n        f"分成 {len(batches)} 批合并。"\n    )\n\n    compacted = []\n\n    for i, batch in enumerate(\n        batches,\n        start=1,\n    ):\n        print(\n            f"  合并批次 {i}/{len(batches)} "\n            f"({len(batch)} issues)..."\n        )\n\n        merged = _merge_one_batch(\n            client,\n            batch,\n            f"第{i}批初步问题",\n        )\n\n        compacted.extend(\n            merged["issues"]\n        )\n\n        print(\n            f"    → {len(merged[\'issues\'])} 个"\n        )\n\n    compacted = (\n        base.pre_dedupe_issues(\n            compacted\n        )\n    )\n\n    if len(compacted) <= (\n        base.MAX_FINAL_ISSUES\n    ):\n        compacted.sort(\n            key=_importance_rank\n        )\n\n        return {\n            "overall_assessment":\n                "已完成分批问题合并。",\n            "issues":\n                compacted[\n                    :base.MAX_FINAL_ISSUES\n                ],\n        }\n\n    print(\n        f"  最终合并：{len(compacted)} → "\n        f"{base.MAX_FINAL_ISSUES} 以内..."\n    )\n\n    final = _merge_one_batch(\n        client,\n        compacted,\n        "分批压缩后的最终候选问题",\n    )\n\n    print(\n        f"合并后保留："\n        f"{len(final[\'issues\'])} 个高价值问题"\n    )\n\n    return final\n\n\n# ============================================================\n# 7. V2.1 主流程：每阶段 checkpoint\n# ============================================================\n\ndef main():\n    api_key = base.require_api_key()\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python review_agent_deepseek_v2_1.py "\n            "\\"你的论文.md\\""\n        )\n        sys.exit(1)\n\n    paper_path, text = (\n        base.read_paper(\n            sys.argv[1]\n        )\n    )\n\n    print("=" * 72)\n    print(\n        "数学建模整篇论文评审 Agent V2.1"\n    )\n    print("=" * 72)\n    print("论文：", paper_path)\n    print("总字符数：", len(text))\n    print("诊断模型：", base.DIAG_MODEL)\n    print("合并模型：", MERGE_MODEL)\n    print(\n        "最终报告模型：",\n        base.REPORT_MODEL,\n    )\n\n    work_dir = make_work_dir(\n        paper_path,\n        text,\n    )\n\n    print(\n        "断点缓存目录：",\n        work_dir,\n    )\n\n    sections = split_sections_v21(\n        text\n    )\n\n    if not sections:\n        raise RuntimeError(\n            "没有识别到任何章节。"\n        )\n\n    outline_text = (\n        base.build_outline(\n            sections\n        )\n    )\n\n    print()\n    print("=" * 72)\n    print("识别到的论文目录")\n    print("=" * 72)\n    print(outline_text)\n\n    save_json(\n        work_dir / "01_outline.json",\n        {\n            "paper": str(paper_path),\n            "sections": sections,\n        },\n    )\n\n    client = base.make_client(\n        api_key\n    )\n\n    # ---------- 1. 局部评审 ----------\n    local_checkpoint = (\n        work_dir\n        / "02_section_reviews_progress.json"\n    )\n\n    local_reviews, local_issues = (\n        review_all_sections_resume(\n            client=client,\n            sections=sections,\n            outline_text=outline_text,\n            checkpoint_path=\n                local_checkpoint,\n        )\n    )\n\n    save_json(\n        work_dir\n        / "02_section_reviews.json",\n        local_reviews,\n    )\n\n    # ---------- 2. 全文结构 ----------\n    global_path = (\n        work_dir\n        / "03_global_review.json"\n    )\n\n    global_review = load_json(\n        global_path\n    )\n\n    if global_review is None:\n        global_review = (\n            review_global_structure_v21(\n                client=client,\n                sections=sections,\n                outline_text=outline_text,\n            )\n        )\n\n        save_json(\n            global_path,\n            global_review,\n        )\n    else:\n        print(\n            "\\n[2/5] 检测到全文结构检查缓存，直接复用。"\n        )\n\n    # ---------- 3. 合并 ----------\n    merged_path = (\n        work_dir\n        / "04_merged_issues.json"\n    )\n\n    merged_data = load_json(\n        merged_path\n    )\n\n    if merged_data is None:\n        merged_data = (\n            merge_issues_v21(\n                client=client,\n                local_issues=\n                    local_issues,\n                global_issues=\n                    global_review[\n                        "issues"\n                    ],\n            )\n        )\n\n        save_json(\n            merged_path,\n            merged_data,\n        )\n    else:\n        print(\n            "\\n[3/5] 检测到问题合并缓存，直接复用。"\n        )\n\n    # ---------- 4. RAG ----------\n    retrieval_path = (\n        work_dir\n        / "05_retrieval.json"\n    )\n\n    enriched_data = load_json(\n        retrieval_path\n    )\n\n    if enriched_data is None:\n        collection, embedding_model = (\n            base.initialize_retrieval()\n        )\n\n        enriched_data = (\n            base.retrieve_for_all_issues(\n                merged_data=merged_data,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        save_json(\n            retrieval_path,\n            enriched_data,\n        )\n    else:\n        print(\n            "\\n[4/5] 检测到 RAG 缓存，直接复用。"\n        )\n\n    # ---------- 5. 最终报告 ----------\n    report_path = (\n        work_dir\n        / "review_report.md"\n    )\n\n    if report_path.exists():\n        final_report = (\n            report_path.read_text(\n                encoding="utf-8"\n            )\n        )\n\n        print(\n            "\\n[5/5] 检测到最终报告缓存，直接复用。"\n        )\n    else:\n        final_report = (\n            base.generate_final_report(\n                client=client,\n                outline_text=\n                    outline_text,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        report_path.write_text(\n            final_report,\n            encoding="utf-8",\n        )\n\n    print()\n    print("=" * 72)\n    print("最终评审报告")\n    print("=" * 72)\n    print()\n    print(final_report)\n\n    print()\n    print("=" * 72)\n    print("本次全部中间结果：")\n    print(work_dir)\n    print("最终 Markdown：")\n    print(report_path)\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2_1 = _load_embedded_module('review_agent_deepseek_v2_1', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2_1)

# ===== Embedded module: review_agent_deepseek_v2_3 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2_3 = '"""\nreview_agent_deepseek_v2_3.py\n\nV2.3 目标：提高评审可信度，而不是增加更多功能。\n\n新增：\n1. Evidence Check\n   - 每个最终 issue 必须重新回到论文正文核验。\n   - unsupported 的问题不会进入最终报告。\n   - 禁止仅凭“篇幅短 / xxx chars”判定问题。\n\n2. RAG 二次重排\n   - 本地向量检索先取最多 10 个跨论文候选。\n   - DeepSeek V4-Flash 判定 strong / structural / weak。\n   - 最终最多保留 3 条，并在结果中记录 relevance 和 reason。\n\n3. 最终报告证据纪律\n   - 只能使用通过 Evidence Check 的问题。\n   - evidence 不足时不猜测。\n   - 章节字符数不能作为问题证据。\n   - 区分“强相关案例”和“结构参考案例”。\n\n4. 缓存\n   - 复用 V2.1/V2.2 的 01~04 阶段。\n   - 新增：\n       04b_verified_issues_v23.json\n       05_retrieval_v23.json\n       review_report_v23.md\n"""\n\nimport json\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n\nimport review_agent_deepseek_v2_1 as v21\n\n\nROOT = Path("D:/math_model_kb")\n\nVERIFY_MODEL = os.getenv(\n    "DEEPSEEK_VERIFY_MODEL",\n    "deepseek-v4-flash",\n)\n\nRERANK_MODEL = os.getenv(\n    "DEEPSEEK_RERANK_MODEL",\n    "deepseek-v4-flash",\n)\n\nREPORT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_MODEL",\n    "deepseek-v4-pro",\n)\n\nRAG_CANDIDATE_K = 10\nRAG_FINAL_K = 3\n\nVERIFY_MAX_CONTEXT = 18000\n\nREPORT_MAX_TOKENS = 30000\nCONTINUE_MAX_TOKENS = 20000\nMAX_CONTINUATIONS = 2\n\n\n# ============================================================\n# 1. JSON 工具\n# ============================================================\n\ndef save_json(path, data):\n    path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path, default=None):\n    if not path.exists():\n        return default\n\n    try:\n        return json.loads(\n            path.read_text(\n                encoding="utf-8"\n            )\n        )\n    except Exception:\n        return default\n\n\n# ============================================================\n# 2. Evidence Check\n# ============================================================\n\nVERIFY_SYSTEM = r"""\n你是数学建模竞赛论文的“证据核验器”。\n\n你现在不是继续找问题，而是核验一个已经被提出的问题是否真的被论文正文支持。\n\n你会看到：\n1. 论文完整目录；\n2. 当前候选问题；\n3. 与该问题最相关的论文原文片段。\n\n原则非常严格：\n\nA. 只有在论文原文或完整目录中存在直接证据时，才能 verdict = "supported"。\n\nB. 如果只是“可能”“看起来”“大概”“很可能”，必须 verdict = "unsupported"。\n\nC. “某章节只有 194 chars / 篇幅较短 / 字数较少”本身不能证明论文有问题。\n   如果候选问题同时包含真正的内容缺失，例如：\n   - 没有定义决策变量；\n   - 没有目标函数；\n   - 没有约束条件；\n   - 参数没有来源；\n   则应删除字数/篇幅论据，只保留真正可以从正文核验的证据。\n\nD. 对“全文缺少某章节/环节”的判断，只能依据完整目录。\n   例如目录中已经存在“灵敏度分析”，就不能说全文缺少灵敏度分析。\n\nE. 不要因为你认为某种模型“应该这样写”就认定原文一定错误。\n   必须区分：\n   - 原文确实数学上缺了必要条件；\n   - 只是可以写得更详细。\n\nF. evidence_from_draft 必须是原文中真实存在的内容或对真实内容的准确概括。\n   不得虚构公式、参数、数字、章节。\n\nG. 不要新增原候选问题之外的新问题。\n\n只输出一个合法 JSON object：\n\n{\n  "verdict": "supported|unsupported",\n  "confidence": 0.0,\n  "problem": "经过证据核验后更精确的问题名称",\n  "reason": "只基于证据的问题原因",\n  "evidence_from_draft": "经过核验后的证据，不使用章节字数作为证据",\n  "evidence_sections": ["相关章节标题"],\n  "verification_note": "为什么通过或未通过核验"\n}\n"""\n\n\ndef normalize_title_for_match(text):\n    return re.sub(\n        r"\\s+",\n        "",\n        text or "",\n    ).lower()\n\n\ndef title_prefixes(title):\n    """\n    从标题提取可能出现在 evidence 中的章节编号。\n    如：\n      5.2 样本平均近似优化 -> 5.2\n      十、结论 -> 十\n    """\n    result = []\n\n    m = re.match(\n        r"^\\s*(\\d+(?:\\.\\d+)*)",\n        title or "",\n    )\n    if m:\n        result.append(\n            m.group(1)\n        )\n\n    m = re.match(\n        r"^\\s*([一二三四五六七八九十]+)[、.．]",\n        title or "",\n    )\n    if m:\n        result.append(\n            m.group(1)\n        )\n\n    return result\n\n\ndef select_evidence_sections(\n    issue,\n    sections,\n):\n    """\n    为一个 issue 找到最值得交给证据核验器看的正文。\n    """\n    source_sections = [\n        normalize_title_for_match(x)\n        for x in issue.get(\n            "source_sections",\n            []\n        )\n        if x\n    ]\n\n    evidence_text = (\n        issue.get(\n            "evidence_from_draft",\n            ""\n        )\n        + "\\n"\n        + issue.get(\n            "reason",\n            ""\n        )\n        + "\\n"\n        + issue.get(\n            "problem",\n            ""\n        )\n    )\n\n    selected = []\n    selected_ids = set()\n\n    def add(sec):\n        sid = sec.get(\n            "section_index"\n        )\n\n        if sid in selected_ids:\n            return\n\n        if sec.get(\n            "section_type"\n        ) in {\n            "front_matter",\n            "reference",\n            "appendix",\n        }:\n            return\n\n        selected.append(sec)\n        selected_ids.add(sid)\n\n    # 1. 优先 source_sections 精确/包含匹配\n    for sec in sections:\n        t = normalize_title_for_match(\n            sec["title"]\n        )\n\n        for source in source_sections:\n            if (\n                source\n                and source != "全文结构"\n                and (\n                    source == t\n                    or source in t\n                    or t in source\n                )\n            ):\n                add(sec)\n\n    # 2. 根据 evidence 中出现的章节号找\n    for sec in sections:\n        for prefix in title_prefixes(\n            sec["title"]\n        ):\n            if re.search(\n                rf"(?<![\\d.]){re.escape(prefix)}(?![\\d.])",\n                evidence_text,\n            ):\n                add(sec)\n\n    # 3. 根据 retrieval_section_type 补相关章节\n    wanted = issue.get(\n        "retrieval_section_type",\n        ""\n    )\n\n    related_map = {\n        "error_analysis": {\n            "validation",\n            "sensitivity",\n            "error_analysis",\n        },\n        "validation": {\n            "validation",\n            "sensitivity",\n            "error_analysis",\n        },\n        "model_selection": {\n            "model_selection",\n            "model_idea",\n        },\n        "model_building": {\n            "model_building",\n            "model_detail",\n        },\n        "model_solving": {\n            "model_solving",\n            "model_detail",\n        },\n        "result": {\n            "result",\n            "validation",\n        },\n        "evaluation": {\n            "evaluation",\n            "strength",\n            "weakness",\n        },\n    }\n\n    wanted_types = related_map.get(\n        wanted,\n        {wanted},\n    )\n\n    for sec in sections:\n        if (\n            sec["section_type"]\n            in wanted_types\n        ):\n            add(sec)\n\n        if len(selected) >= 8:\n            break\n\n    # 4. 如果还是完全没找到，至少选正文最相关的前几个非空章节\n    if not selected:\n        for sec in sections:\n            if (\n                sec["char_count"] >= 80\n                and sec["section_type"]\n                not in {\n                    "front_matter",\n                    "reference",\n                    "appendix",\n                }\n            ):\n                add(sec)\n\n            if len(selected) >= 5:\n                break\n\n    return selected\n\n\ndef build_evidence_context(\n    issue,\n    outline_text,\n    sections,\n):\n    selected = (\n        select_evidence_sections(\n            issue,\n            sections,\n        )\n    )\n\n    blocks = []\n\n    blocks.append(\n        "===== 完整论文目录 =====\\n"\n        + outline_text\n    )\n\n    blocks.append(\n        "\\n===== 候选问题 =====\\n"\n        + json.dumps(\n            issue,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    blocks.append(\n        "\\n===== 相关论文原文 ====="\n    )\n\n    total = 0\n\n    for sec in selected:\n        content = sec[\n            "content"\n        ].strip()\n\n        block = (\n            f"\\n\\n### {sec[\'title\']}\\n"\n            f"{content}\\n"\n        )\n\n        if (\n            total + len(block)\n            > VERIFY_MAX_CONTEXT\n        ):\n            remaining = (\n                VERIFY_MAX_CONTEXT\n                - total\n            )\n\n            if remaining > 500:\n                blocks.append(\n                    block[:remaining]\n                )\n\n            break\n\n        blocks.append(block)\n        total += len(block)\n\n    return "\\n".join(blocks)\n\n\ndef verify_one_issue(\n    client,\n    issue,\n    sections,\n    outline_text,\n):\n    prompt = (\n        build_evidence_context(\n            issue=issue,\n            outline_text=\n                outline_text,\n            sections=sections,\n        )\n    )\n\n    data = (\n        v21.reliable_call_json(\n            client=client,\n            model=VERIFY_MODEL,\n            system_prompt=VERIFY_SYSTEM,\n            user_prompt=prompt,\n            max_tokens=3500,\n            retries=3,\n        )\n    )\n\n    verdict = data.get(\n        "verdict",\n        "unsupported",\n    )\n\n    try:\n        confidence = float(\n            data.get(\n                "confidence",\n                0,\n            )\n        )\n    except Exception:\n        confidence = 0.0\n\n    supported = (\n        verdict == "supported"\n        and confidence >= 0.70\n    )\n\n    result = dict(issue)\n\n    result[\n        "evidence_verification"\n    ] = {\n        "verdict": verdict,\n        "confidence": confidence,\n        "verification_note":\n            data.get(\n                "verification_note",\n                "",\n            ),\n        "evidence_sections":\n            data.get(\n                "evidence_sections",\n                [],\n            ),\n    }\n\n    if supported:\n        # 用核验后的、更谨慎的文本覆盖旧文本\n        result["problem"] = (\n            data.get(\n                "problem"\n            )\n            or issue.get(\n                "problem",\n                ""\n            )\n        )\n\n        result["reason"] = (\n            data.get(\n                "reason"\n            )\n            or issue.get(\n                "reason",\n                ""\n            )\n        )\n\n        result[\n            "evidence_from_draft"\n        ] = (\n            data.get(\n                "evidence_from_draft"\n            )\n            or issue.get(\n                "evidence_from_draft",\n                ""\n            )\n        )\n\n        result[\n            "source_sections"\n        ] = (\n            data.get(\n                "evidence_sections"\n            )\n            or issue.get(\n                "source_sections",\n                []\n            )\n        )\n\n    return supported, result\n\n\ndef verify_all_issues(\n    client,\n    merged_data,\n    sections,\n    outline_text,\n):\n    print(\n        "\\n[3.5/5] 正在做 Evidence Check..."\n    )\n\n    accepted = []\n    rejected = []\n\n    issues = merged_data.get(\n        "issues",\n        [],\n    )\n\n    for i, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        print(\n            f"\\n[{i}/{len(issues)}] "\n            f"{issue.get(\'problem\')}"\n        )\n\n        supported, result = (\n            verify_one_issue(\n                client=client,\n                issue=issue,\n                sections=sections,\n                outline_text=\n                    outline_text,\n            )\n        )\n\n        verdict_info = result[\n            "evidence_verification"\n        ]\n\n        print(\n            "Evidence：",\n            verdict_info[\n                "verdict"\n            ],\n            " confidence=",\n            verdict_info[\n                "confidence"\n            ],\n        )\n\n        if supported:\n            accepted.append(result)\n        else:\n            rejected.append(result)\n\n    print()\n    print(\n        f"Evidence Check 完成："\n        f"通过 {len(accepted)}，"\n        f"淘汰 {len(rejected)}。"\n    )\n\n    return {\n        "overall_assessment":\n            merged_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues": accepted,\n        "rejected_issues":\n            rejected,\n    }\n\n\n# ============================================================\n# 3. RAG Top10 + LLM rerank\n# ============================================================\n\nRERANK_SYSTEM = r"""\n你是数学建模优秀论文案例的“检索相关性重排器”。\n\n你会看到：\n1. 当前论文中的一个已通过证据核验的问题；\n2. 用户希望检索的 section_type 和 query；\n3. 从优秀论文知识库向量检索得到的若干候选案例。\n\n请判断每个候选案例对当前问题的参考价值。\n\n相关性定义：\n\nstrong：\n- 候选案例直接处理了与当前问题相同或高度相似的建模/验证/写作问题；\n- 或直接展示了当前问题所缺失的方法。\n例如：\n当前问题是“SAA 没有数学形式”，候选案例本身也明确展示随机优化、样本均值近似、期望目标函数等。\n\nstructural：\n- 具体模型不同，但能清楚展示当前论文应采用的结构或表达方式。\n例如：\n当前问题是“约束条件不完整”，候选论文虽然是无人机模型，但完整地展示了“逐项定义约束 + 物理含义”，可作为结构参考。\n\nweak：\n- 只有非常泛的关键词重叠；\n- 只因为都叫“模型求解/结果分析”而相似；\n- 无法给当前问题提供直接方法或清晰结构参考。\n\n规则：\n1. 不要因为论文是“优秀论文”就默认 strong。\n2. 优先 strong，其次 structural。\n3. weak 不应进入最终报告。\n4. 不允许改变候选内容的原意。\n5. 只输出 JSON：\n\n{\n  "ranking": [\n    {\n      "candidate_id": 1,\n      "relevance": "strong|structural|weak",\n      "score": 0.0,\n      "reason": "为什么与当前问题相关/不相关"\n    }\n  ]\n}\n"""\n\n\ndef collect_vector_candidates(\n    issue,\n    embedding_model,\n    collection,\n):\n    primary = issue[\n        "retrieval_section_type"\n    ]\n\n    desired_types = [primary]\n\n    fallback_types = (\n        v21.base\n        .FALLBACK_SECTION_TYPES\n        .get(primary, [])\n    )\n\n    rows, _ = (\n        v21.base.query_candidates(\n            model=embedding_model,\n            collection=collection,\n            query=issue[\n                "retrieval_query"\n            ],\n            section_types=\n                desired_types,\n            candidate_k=80,\n        )\n    )\n\n    selected, _ = (\n        v21.base.rerank_results(\n            rows=rows,\n            section_types=\n                desired_types,\n            top_k=\n                RAG_CANDIDATE_K,\n        )\n    )\n\n    used_papers = {\n        row["metadata"].get(\n            "paper_id"\n        )\n        for row in selected\n    }\n\n    # primary 不足 10 个时用 fallback 补\n    for fallback in fallback_types:\n        if (\n            len(selected)\n            >= RAG_CANDIDATE_K\n        ):\n            break\n\n        extra_rows, _ = (\n            v21.base.query_candidates(\n                model=embedding_model,\n                collection=collection,\n                query=issue[\n                    "retrieval_query"\n                ],\n                section_types=[\n                    fallback\n                ],\n                candidate_k=50,\n            )\n        )\n\n        extras, _ = (\n            v21.base.rerank_results(\n                rows=extra_rows,\n                section_types=[\n                    fallback\n                ],\n                top_k=\n                    RAG_CANDIDATE_K,\n            )\n        )\n\n        for row in extras:\n            pid = row[\n                "metadata"\n            ].get(\n                "paper_id"\n            )\n\n            if pid in used_papers:\n                continue\n\n            selected.append(row)\n            used_papers.add(pid)\n\n            if (\n                len(selected)\n                >= RAG_CANDIDATE_K\n            ):\n                break\n\n    return selected[\n        :RAG_CANDIDATE_K\n    ]\n\n\ndef candidate_to_dict(\n    row,\n    candidate_id,\n):\n    meta = row["metadata"]\n\n    return {\n        "candidate_id":\n            candidate_id,\n        "paper_id":\n            meta.get("paper_id"),\n        "problem_id":\n            meta.get(\n                "problem_id"\n            ),\n        "question_number":\n            meta.get(\n                "question_number"\n            ),\n        "section_type":\n            meta.get(\n                "section_type"\n            ),\n        "section_title":\n            meta.get(\n                "section_title"\n            ),\n        "chunk_id":\n            row["chunk_id"],\n        "distance":\n            round(\n                float(\n                    row["distance"]\n                ),\n                4,\n            ),\n        "content":\n            v21.base.extract_body(\n                row["document"]\n            )[:1800],\n    }\n\n\ndef rerank_candidates_with_llm(\n    client,\n    issue,\n    candidates,\n):\n    if not candidates:\n        return []\n\n    candidate_dicts = [\n        candidate_to_dict(\n            row,\n            i,\n        )\n        for i, row in enumerate(\n            candidates,\n            start=1,\n        )\n    ]\n\n    prompt = (\n        "===== 当前论文问题 =====\\n"\n        f"problem: {issue.get(\'problem\')}\\n"\n        f"reason: {issue.get(\'reason\')}\\n"\n        f"evidence: {issue.get(\'evidence_from_draft\')}\\n"\n        f"retrieval_section_type: "\n        f"{issue.get(\'retrieval_section_type\')}\\n"\n        f"retrieval_query: "\n        f"{issue.get(\'retrieval_query\')}\\n\\n"\n        "===== 候选优秀论文案例 =====\\n"\n        + json.dumps(\n            candidate_dicts,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    data = (\n        v21.reliable_call_json(\n            client=client,\n            model=RERANK_MODEL,\n            system_prompt=\n                RERANK_SYSTEM,\n            user_prompt=prompt,\n            max_tokens=4500,\n            retries=3,\n        )\n    )\n\n    ranking_map = {}\n\n    for item in data.get(\n        "ranking",\n        [],\n    ):\n        try:\n            cid = int(\n                item.get(\n                    "candidate_id"\n                )\n            )\n        except Exception:\n            continue\n\n        relevance = item.get(\n            "relevance",\n            "weak",\n        )\n\n        if relevance not in {\n            "strong",\n            "structural",\n            "weak",\n        }:\n            relevance = "weak"\n\n        try:\n            score = float(\n                item.get(\n                    "score",\n                    0,\n                )\n            )\n        except Exception:\n            score = 0.0\n\n        ranking_map[cid] = {\n            "relevance":\n                relevance,\n            "score": score,\n            "reason":\n                item.get(\n                    "reason",\n                    "",\n                ),\n        }\n\n    enriched = []\n\n    for item in candidate_dicts:\n        cid = item[\n            "candidate_id"\n        ]\n\n        rank = ranking_map.get(\n            cid,\n            {\n                "relevance":\n                    "weak",\n                "score": 0.0,\n                "reason":\n                    "LLM 未返回该候选的相关性判断。",\n            },\n        )\n\n        x = dict(item)\n        x.update(rank)\n        enriched.append(x)\n\n    priority = {\n        "strong": 0,\n        "structural": 1,\n        "weak": 2,\n    }\n\n    enriched.sort(\n        key=lambda x: (\n            priority[\n                x["relevance"]\n            ],\n            -x["score"],\n            x["distance"],\n        )\n    )\n\n    # weak 完全不进入最终结果\n    usable = [\n        x for x in enriched\n        if x["relevance"]\n        in {\n            "strong",\n            "structural",\n        }\n    ]\n\n    return usable[\n        :RAG_FINAL_K\n    ]\n\n\ndef retrieve_v23(\n    client,\n    verified_data,\n    embedding_model,\n    collection,\n):\n    print(\n        "\\n[4/5] 正在执行 "\n        "Top10 检索 + DeepSeek rerank..."\n    )\n\n    result_issues = []\n\n    issues = verified_data.get(\n        "issues",\n        [],\n    )\n\n    for i, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        print(\n            f"\\n[{i}/{len(issues)}] "\n            f"{issue.get(\'problem\')}"\n        )\n\n        vector_candidates = (\n            collect_vector_candidates(\n                issue=issue,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        print(\n            "向量候选：",\n            len(\n                vector_candidates\n            ),\n        )\n\n        final_examples = (\n            rerank_candidates_with_llm(\n                client=client,\n                issue=issue,\n                candidates=\n                    vector_candidates,\n            )\n        )\n\n        strong_n = sum(\n            1\n            for x in final_examples\n            if x[\n                "relevance"\n            ] == "strong"\n        )\n\n        structural_n = sum(\n            1\n            for x in final_examples\n            if x[\n                "relevance"\n            ] == "structural"\n        )\n\n        print(\n            f"最终案例："\n            f"{len(final_examples)} "\n            f"(strong={strong_n}, "\n            f"structural={structural_n})"\n        )\n\n        x = dict(issue)\n\n        x["retrieval"] = {\n            "query":\n                issue[\n                    "retrieval_query"\n                ],\n            "primary_section_type":\n                issue[\n                    "retrieval_section_type"\n                ],\n            "vector_candidate_count":\n                len(\n                    vector_candidates\n                ),\n            "examples":\n                final_examples,\n        }\n\n        result_issues.append(x)\n\n    return {\n        "overall_assessment":\n            verified_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues":\n            result_issues,\n        "rejected_issues":\n            verified_data.get(\n                "rejected_issues",\n                [],\n            ),\n    }\n\n\n# ============================================================\n# 4. V2.3 最终报告\n# ============================================================\n\nFINAL_REPORT_SYSTEM_V23 = r"""\n你是一名数学建模竞赛论文指导型评审助手。\n\n你收到的问题全部已经过 Evidence Check。\n因此你只能基于这些“已核验问题”写最终报告，不得新增未经核验的问题。\n\n你还会收到优秀论文案例。每个案例已经被二次相关性重排：\n- strong：与当前问题直接高度相关；\n- structural：模型不同，但可以作为结构/表达方式参考。\n\n必须遵守：\n\n1. 不得猜测。\n如果证据没有展示摘要存在某问题，就不能写“摘要很可能也有这个问题”。\n\n2. 禁止把：\n- 章节字符数；\n- 篇幅短；\n- 字数少\n作为问题成立的证据。\n只能说正文具体缺了什么。\n\n3. strong 案例可以重点参考其方法。\nstructural 案例必须明确写成：\n“模型场景不同，这里仅参考其……结构/表达方式”。\n\n4. 不得把知识库案例中的具体模型机械套用到当前论文。\n\n5. 不得没有 Rubric 就给：\n- 70%-75%\n- 80分\n- A等级\n等伪精确总评分。\n\n6. 最优先的问题应优先排序：\n数学错误/不可行性风险\n> 模型定义完整性\n> 参数依据与可复现性\n> 验证与结果可信度\n> 表达与总结优化。\n\n7. 每个问题包含：\n- 问题是什么\n- 为什么重要\n- 草稿中的核验证据\n- 优秀论文案例怎么处理\n- 从案例抽象什么\n- 作者如何修改\n\n8. 如果某 issue 没有留下 strong/structural 案例，也可以正常给修改建议，\n但必须明确说“知识库本次没有检索到足够强的同类案例”，不要硬凑引用。\n\n完整输出：\n\n# 一、总体评价\n\n# 二、最优先修改的问题\n\n# 三、分模块修改建议\n\n# 四、修改顺序\n\n最后必须有：\nP0：必须优先改\nP1：重点改\nP2：有时间再优化\n"""\n\n\ndef build_v23_reference_text(\n    enriched_data,\n):\n    blocks = []\n\n    for i, issue in enumerate(\n        enriched_data.get(\n            "issues",\n            [],\n        ),\n        start=1,\n    ):\n        verify = issue.get(\n            "evidence_verification",\n            {},\n        )\n\n        blocks.append(\n            f"## Issue {i}: "\n            f"{issue.get(\'problem\')}"\n        )\n        blocks.append(\n            f"importance: "\n            f"{issue.get(\'importance\')}"\n        )\n        blocks.append(\n            f"evidence_verified: "\n            f"{verify.get(\'verdict\')} "\n            f"(confidence="\n            f"{verify.get(\'confidence\')})"\n        )\n        blocks.append(\n            "source_sections: "\n            + ", ".join(\n                issue.get(\n                    "source_sections",\n                    [],\n                )\n            )\n        )\n        blocks.append(\n            "reason: "\n            + issue.get(\n                "reason",\n                "",\n            )\n        )\n        blocks.append(\n            "evidence: "\n            + issue.get(\n                "evidence_from_draft",\n                "",\n            )\n        )\n        blocks.append(\n            "suggested_direction: "\n            + issue.get(\n                "suggested_direction",\n                "",\n            )\n        )\n\n        examples = (\n            issue.get(\n                "retrieval",\n                {}\n            ).get(\n                "examples",\n                [],\n            )\n        )\n\n        if not examples:\n            blocks.append(\n                "RAG examples: none"\n            )\n            continue\n\n        for j, ex in enumerate(\n            examples,\n            start=1,\n        ):\n            blocks.append(\n                f"\\nExample {j}\\n"\n                f"relevance: "\n                f"{ex[\'relevance\']}\\n"\n                f"rerank_score: "\n                f"{ex[\'score\']}\\n"\n                f"rerank_reason: "\n                f"{ex[\'reason\']}\\n"\n                f"paper_id: "\n                f"{ex[\'paper_id\']}\\n"\n                f"section_title: "\n                f"{ex[\'section_title\']}\\n"\n                f"section_type: "\n                f"{ex[\'section_type\']}\\n"\n                f"content:\\n"\n                f"{ex[\'content\']}\\n"\n            )\n\n    return "\\n".join(blocks)\n\n\ndef report_complete(\n    text,\n):\n    required = [\n        "一、总体评价",\n        "二、最优先修改的问题",\n        "三、分模块修改建议",\n        "四、修改顺序",\n        "P0",\n        "P1",\n        "P2",\n    ]\n\n    return (\n        bool(text)\n        and all(\n            x in text\n            for x in required\n        )\n    )\n\n\ndef call_report(\n    client,\n    messages,\n    max_tokens,\n):\n    response = (\n        client.chat.completions.create(\n            model=REPORT_MODEL,\n            messages=messages,\n            temperature=0.25,\n            max_tokens=max_tokens,\n            extra_body={\n                "thinking": {\n                    "type":\n                        "disabled"\n                }\n            },\n        )\n    )\n\n    choice = response.choices[0]\n\n    content = (\n        choice.message.content\n        or ""\n    ).strip()\n\n    return (\n        content,\n        getattr(\n            choice,\n            "finish_reason",\n            None,\n        ),\n    )\n\n\ndef generate_report_v23(\n    client,\n    outline_text,\n    enriched_data,\n):\n    print(\n        f"\\n[5/5] 正在用 "\n        f"{REPORT_MODEL} "\n        "生成 V2.3 最终报告..."\n    )\n\n    reference_text = (\n        build_v23_reference_text(\n            enriched_data\n        )\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline_text}\\n\\n"\n        "===== 已通过 Evidence Check "\n        "的问题 + rerank 后优秀论文案例 =====\\n"\n        f"{reference_text}\\n\\n"\n        "只评审上述已经通过证据核验的问题。"\n    )\n\n    messages = [\n        {\n            "role": "system",\n            "content":\n                FINAL_REPORT_SYSTEM_V23,\n        },\n        {\n            "role": "user",\n            "content":\n                user_prompt,\n        },\n    ]\n\n    content, finish_reason = (\n        call_report(\n            client,\n            messages,\n            REPORT_MAX_TOKENS,\n        )\n    )\n\n    if not content:\n        raise RuntimeError(\n            "V2.3 最终报告返回空内容"\n        )\n\n    full = content\n\n    print(\n        "报告 finish_reason：",\n        finish_reason,\n    )\n\n    count = 0\n\n    while (\n        finish_reason == "length"\n        and count\n        < MAX_CONTINUATIONS\n    ):\n        count += 1\n\n        print(\n            f"报告被截断，自动续写 "\n            f"{count}/{MAX_CONTINUATIONS}"\n        )\n\n        messages.append({\n            "role": "assistant",\n            "content": full,\n        })\n        messages.append({\n            "role": "user",\n            "content":\n                "从截断位置继续，"\n                "不要重复已写内容，"\n                "最终必须完成"\n                "“三、分模块修改建议”、"\n                "“四、修改顺序”以及 P0/P1/P2。",\n        })\n\n        cont, finish_reason = (\n            call_report(\n                client,\n                messages,\n                CONTINUE_MAX_TOKENS,\n            )\n        )\n\n        if not cont:\n            break\n\n        full = (\n            full.rstrip()\n            + "\\n\\n"\n            + cont.lstrip()\n        )\n\n    if not report_complete(\n        full\n    ):\n        print(\n            "警告：V2.3 报告完整性检查未通过。"\n        )\n\n    return full.strip()\n\n\n# ============================================================\n# 5. 主程序\n# ============================================================\n\ndef main():\n    api_key = (\n        v21.base.require_api_key()\n    )\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python review_agent_deepseek_v2_3.py "\n            "\\"你的论文.md\\""\n        )\n        sys.exit(1)\n\n    paper_path, text = (\n        v21.base.read_paper(\n            sys.argv[1]\n        )\n    )\n\n    print("=" * 72)\n    print(\n        "数学建模整篇论文评审 Agent V2.3"\n    )\n    print("=" * 72)\n    print("论文：", paper_path)\n    print("总字符数：", len(text))\n    print(\n        "Evidence模型：",\n        VERIFY_MODEL,\n    )\n    print(\n        "Rerank模型：",\n        RERANK_MODEL,\n    )\n    print(\n        "报告模型：",\n        REPORT_MODEL,\n    )\n\n    work_dir = (\n        v21.make_work_dir(\n            paper_path,\n            text,\n        )\n    )\n\n    print(\n        "工作目录：",\n        work_dir,\n    )\n\n    sections = (\n        v21.split_sections_v21(\n            text\n        )\n    )\n\n    outline_text = (\n        v21.base.build_outline(\n            sections\n        )\n    )\n\n    client = (\n        v21.base.make_client(\n            api_key\n        )\n    )\n\n    # --------------------------------------------------------\n    # Stage 1/2/3:\n    # 优先复用 V2.1/V2.2 已有缓存\n    # --------------------------------------------------------\n\n    local_path = (\n        work_dir\n        / "02_section_reviews.json"\n    )\n\n    global_path = (\n        work_dir\n        / "03_global_review.json"\n    )\n\n    merged_path = (\n        work_dir\n        / "04_merged_issues.json"\n    )\n\n    local_reviews = load_json(\n        local_path\n    )\n\n    if local_reviews is None:\n        print(\n            "\\n未发现局部评审缓存，"\n            "需要执行第 1 阶段。"\n        )\n\n        progress_path = (\n            work_dir\n            / "02_section_reviews_progress.json"\n        )\n\n        local_reviews, local_issues = (\n            v21.review_all_sections_resume(\n                client=client,\n                sections=sections,\n                outline_text=\n                    outline_text,\n                checkpoint_path=\n                    progress_path,\n            )\n        )\n\n        save_json(\n            local_path,\n            local_reviews,\n        )\n    else:\n        print(\n            "\\n[1/5] 检测到局部评审缓存，直接复用。"\n        )\n\n        local_issues = []\n\n        for sec_result in (\n            local_reviews\n        ):\n            for part in sec_result.get(\n                "parts",\n                [],\n            ):\n                local_issues.extend(\n                    part.get(\n                        "issues",\n                        [],\n                    )\n                )\n\n    global_review = load_json(\n        global_path\n    )\n\n    if global_review is None:\n        global_review = (\n            v21.review_global_structure_v21(\n                client=client,\n                sections=sections,\n                outline_text=\n                    outline_text,\n            )\n        )\n        save_json(\n            global_path,\n            global_review,\n        )\n    else:\n        print(\n            "[2/5] 检测到全文结构缓存，直接复用。"\n        )\n\n    merged_data = load_json(\n        merged_path\n    )\n\n    if merged_data is None:\n        merged_data = (\n            v21.merge_issues_v21(\n                client=client,\n                local_issues=\n                    local_issues,\n                global_issues=\n                    global_review[\n                        "issues"\n                    ],\n            )\n        )\n        save_json(\n            merged_path,\n            merged_data,\n        )\n    else:\n        print(\n            "[3/5] 检测到问题合并缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Stage 3.5 Evidence Check\n    # --------------------------------------------------------\n\n    verified_path = (\n        work_dir\n        / "04b_verified_issues_v23.json"\n    )\n\n    verified_data = load_json(\n        verified_path\n    )\n\n    if verified_data is None:\n        verified_data = (\n            verify_all_issues(\n                client=client,\n                merged_data=\n                    merged_data,\n                sections=sections,\n                outline_text=\n                    outline_text,\n            )\n        )\n\n        save_json(\n            verified_path,\n            verified_data,\n        )\n    else:\n        print(\n            "[3.5/5] 检测到 Evidence Check "\n            "缓存，直接复用。"\n        )\n\n    if not verified_data.get(\n        "issues"\n    ):\n        print(\n            "\\n没有问题通过 Evidence Check。"\n        )\n        print(\n            "这意味着当前合并问题证据不足，"\n            "程序不会强行生成批评。"\n        )\n        return\n\n    # --------------------------------------------------------\n    # Stage 4 RAG + rerank\n    # --------------------------------------------------------\n\n    retrieval_path = (\n        work_dir\n        / "05_retrieval_v23.json"\n    )\n\n    enriched_data = load_json(\n        retrieval_path\n    )\n\n    if enriched_data is None:\n        collection, embedding_model = (\n            v21.base.initialize_retrieval()\n        )\n\n        enriched_data = (\n            retrieve_v23(\n                client=client,\n                verified_data=\n                    verified_data,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        save_json(\n            retrieval_path,\n            enriched_data,\n        )\n    else:\n        print(\n            "[4/5] 检测到 V2.3 RAG "\n            "缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Stage 5 final report\n    # --------------------------------------------------------\n\n    report_path = (\n        work_dir\n        / "review_report_v23.md"\n    )\n\n    if report_path.exists():\n        report = (\n            report_path.read_text(\n                encoding="utf-8",\n                errors="ignore",\n            )\n        )\n\n        if report_complete(\n            report\n        ):\n            print(\n                "[5/5] 检测到完整 V2.3 报告，直接复用。"\n            )\n        else:\n            backup = (\n                work_dir\n                / "review_report_v23_incomplete_backup.md"\n            )\n\n            backup.write_text(\n                report,\n                encoding="utf-8",\n            )\n\n            report = (\n                generate_report_v23(\n                    client=client,\n                    outline_text=\n                        outline_text,\n                    enriched_data=\n                        enriched_data,\n                )\n            )\n\n            report_path.write_text(\n                report,\n                encoding="utf-8",\n            )\n    else:\n        report = (\n            generate_report_v23(\n                client=client,\n                outline_text=\n                    outline_text,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        report_path.write_text(\n            report,\n            encoding="utf-8",\n        )\n\n    print()\n    print("=" * 72)\n    print("V2.3 最终评审报告")\n    print("=" * 72)\n    print()\n    print(report)\n\n    print()\n    print("=" * 72)\n    print(\n        "Evidence Check：",\n        verified_path,\n    )\n    print(\n        "RAG rerank：",\n        retrieval_path,\n    )\n    print(\n        "最终报告：",\n        report_path,\n    )\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2_3 = _load_embedded_module('review_agent_deepseek_v2_3', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2_3)

# ===== Embedded module: review_agent_deepseek_v2_4 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2_4 = '"""\nreview_agent_deepseek_v2_4.py\n\nV2.4：修复 V2.3 Evidence Check 过度保守的问题。\n\n核心变化：\n1. Evidence verdict:\n   supported / refined / unsupported\n\n2. 支持“缺失证据”：\n   当完整目标章节已经提供时，章节中没有出现必要的数学要素，\n   可以作为 absence_in_complete_scope 证据。\n\n3. 不再 all-or-nothing：\n   原问题过重但核心问题成立时，自动 refined 为更精确的问题。\n   例如：\n   - “多个参数都未定义” -> “个别参数定义/计算方式仍不完整”\n   - “SAA目标和变量未明确” -> “SAA有文字目标，但缺少正式数学表达和约束”\n\n4. Evidence Check 重新评 importance：\n   high   = 数学错误/不可行性/核心模型无法复现\n   medium = 可信度/验证/形式化明显不足\n   low    = 呈现与总结优化\n\n5. 不允许“字符数/篇幅短”进入证据或最终报告。\n\n6. 复用 V2.1/V2.2 的 01~04 缓存；\n   重新执行 V2.4 Evidence + V2.3 RAG rerank + final report。\n"""\n\nimport json\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n\nimport review_agent_deepseek_v2_1 as v21\nimport review_agent_deepseek_v2_3 as v23\n\n\nVERIFY_MODEL = os.getenv(\n    "DEEPSEEK_VERIFY_MODEL",\n    "deepseek-v4-flash",\n)\n\nREPORT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_MODEL",\n    "deepseek-v4-pro",\n)\n\nVERIFY_MAX_CONTEXT = 26000\n\nREPORT_MAX_TOKENS = 30000\nCONTINUE_MAX_TOKENS = 20000\nMAX_CONTINUATIONS = 2\n\n\nVERIFY_SYSTEM_V24 = r"""\n你是数学建模竞赛论文的“证据核验与问题校准器”。\n\n你的任务不是简单地把问题删掉，而是判断：\n- 原问题是否成立；\n- 如果原问题说得太重，是否应该收窄成一个更精确、可证实的问题。\n\n你会看到：\n1. 完整论文目录；\n2. 一个候选问题；\n3. 与该问题有关的若干【完整章节正文】。\n\n====================\n一、verdict 三种\n====================\n\nsupported：\n原候选问题的核心表述可以被证据直接支持。\n\nrefined：\n原候选问题说得过重、过宽或措辞不准确，\n但从完整章节中仍然能确认一个更窄、更准确的问题。\n此时不要整条删除，而要重写 problem / reason / evidence。\n\nunsupported：\n原文与候选问题矛盾，或现有完整章节不足以证明任何有价值的问题。\n\n====================\n二、缺失类问题如何核验\n====================\n\n特别重要：\n\n“缺少某要素”不要求论文作者在原文中写一句“我缺少某要素”。\n\n如果系统明确告诉你提供的是【完整章节正文】，\n那么你可以检查这个完整章节中是否存在目标要素。\n\n例如：\n\nA. 完整 5.1 节只给出了：\n0 <= q_ji <= min(...)\n且该模型允许一个舱位与多个舱位发生转移，\n但完整章节中没有任何按 i/j 求和的共享容量约束。\n那么“缺少聚合容量约束”可以由\nabsence_in_complete_scope 支持。\n\nB. 完整 SAA 章节只写：\n“生成 N 组样本，对候选方案比较平均利润”。\n这已经说明了优化目标的语义和决策对象，\n因此不能说“完全没有目标/决策变量”；\n但如果没有写出样本均值目标函数、可行域/约束、Profit定义，\n可以 refined 为：\n“SAA 的正式数学表达与约束说明不足”。\n\nC. 如果符号表已经定义了大部分参数，\n不能继续说“多个参数均未定义”。\n但若仍有个别关键参数没有定义、计算式或取值依据，\n应 refined 为更窄的问题，而不是全部删除。\n\n====================\n三、问题类型与重要性\n====================\n\nissue_kind 取：\n- mathematical_error\n- model_completeness\n- parameter_basis\n- reproducibility\n- validation_gap\n- presentation\n\nimportance 重新判断：\n\nhigh：\n- 可能产生数学不可行解；\n- 目标函数/必要约束等核心模型要素确实缺失；\n- 参数无依据且直接决定核心结论；\n- 会实质影响模型正确性。\n\nmedium：\n- 模型概念存在，但形式化/求解/验证细节不足；\n- 可复现性明显不足；\n- 验证指标不足；\n- 不一定导致模型错误，但影响可信度。\n\nlow：\n- 结论组织、表格化、逻辑串联、表达优化等；\n- 不影响模型正确性。\n\n不要沿用候选问题原来的 high，\n必须重新判断。\n\n====================\n四、禁止事项\n====================\n\n1. “132 chars / 194 chars / 篇幅短 / 字数少”\n   不能作为问题证据，也不能出现在 evidence_from_draft。\n\n2. 没有单独设置某个章节标题，\n   不等于缺少该内容。\n   例如没有“误差分析”标题，但正文已有足够误差验证，\n   则不能批评“全文缺少误差分析”。\n\n3. 不要因为某种写法“更漂亮”就判 high。\n\n4. 不得虚构原文中没有的公式、数字或章节。\n\n====================\n五、输出\n====================\n\n只输出合法 JSON object：\n\n{\n  "verdict": "supported|refined|unsupported",\n  "confidence": 0.0,\n  "issue_kind": "mathematical_error|model_completeness|parameter_basis|reproducibility|validation_gap|presentation",\n  "importance": "high|medium|low",\n  "problem": "核验后最终的问题名称；refined 时必须重写得更精确",\n  "reason": "为什么构成问题",\n  "evidence_from_draft": "只写真正的正文证据，不写字符数/篇幅",\n  "evidence_sections": ["章节标题"],\n  "evidence_basis": "direct|absence_in_complete_scope|mixed",\n  "scope_sufficient": true,\n  "retrieval_section_type": "从原候选中保留或修正",\n  "retrieval_query": "针对最终问题重新生成的知识库检索语句",\n  "suggested_direction": "修改方向",\n  "verification_note": "解释为什么 supported/refined/unsupported"\n}\n"""\n\n\ndef save_json(path, data):\n    path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path, default=None):\n    if not path.exists():\n        return default\n    try:\n        return json.loads(\n            path.read_text(\n                encoding="utf-8"\n            )\n        )\n    except Exception:\n        return default\n\n\ndef clean_title(text):\n    return re.sub(\n        r"\\s+",\n        "",\n        text or "",\n    ).lower()\n\n\ndef issue_keywords(issue):\n    text = " ".join([\n        issue.get("problem", ""),\n        issue.get("reason", ""),\n        issue.get(\n            "evidence_from_draft",\n            "",\n        ),\n        issue.get(\n            "retrieval_query",\n            "",\n        ),\n    ])\n\n    words = re.findall(\n        r"[\\u4e00-\\u9fff]{2,}|[A-Za-z]{2,}|\\d+(?:\\.\\d+)+",\n        text,\n    )\n\n    stop = {\n        "问题",\n        "模型",\n        "论文",\n        "结果",\n        "数学建模",\n        "缺少",\n        "不足",\n        "分析",\n        "说明",\n    }\n\n    return [\n        x.lower()\n        for x in words\n        if x not in stop\n    ][:30]\n\n\ndef exact_source_match(\n    source_title,\n    sec_title,\n):\n    a = clean_title(source_title)\n    b = clean_title(sec_title)\n\n    if not a or not b:\n        return False\n\n    if a == "全文结构":\n        return False\n\n    return (\n        a == b\n        or a in b\n        or b in a\n    )\n\n\ndef choose_complete_scope(\n    issue,\n    sections,\n):\n    """\n    选择完整章节。\n    与 V2.3 的区别：\n    - source_sections 必须完整提供；\n    - 根据问题内容补 notation / validation 等必要交叉章节；\n    - 对全文问题使用相关章节集合。\n    """\n    selected = []\n    ids = set()\n\n    def add(sec):\n        sid = sec.get(\n            "section_index"\n        )\n\n        if sid in ids:\n            return\n\n        if sec.get(\n            "section_type"\n        ) in {\n            "front_matter",\n            "reference",\n            "appendix",\n        }:\n            return\n\n        selected.append(sec)\n        ids.add(sid)\n\n    sources = issue.get(\n        "source_sections",\n        [],\n    )\n\n    # 1. 原 issue 指定的章节全部加入\n    for source in sources:\n        for sec in sections:\n            if exact_source_match(\n                source,\n                sec["title"],\n            ):\n                add(sec)\n\n    # 2. 根据章节号在 evidence 中补充\n    evidence = (\n        issue.get(\n            "evidence_from_draft",\n            ""\n        )\n        + " "\n        + issue.get(\n            "reason",\n            ""\n        )\n    )\n\n    for sec in sections:\n        m = re.match(\n            r"^\\s*(\\d+(?:\\.\\d+)*)",\n            sec["title"],\n        )\n        if m:\n            num = m.group(1)\n            if re.search(\n                rf"(?<![\\d.]){re.escape(num)}(?![\\d.])",\n                evidence,\n            ):\n                add(sec)\n\n    # 3. 特殊交叉核验\n    problem_text = (\n        issue.get("problem", "")\n        + " "\n        + issue.get("reason", "")\n    )\n\n    if any(x in problem_text for x in [\n        "参数定义",\n        "符号",\n        "未定义",\n    ]):\n        for sec in sections:\n            if sec["section_type"] == "notation":\n                add(sec)\n\n    if any(x in problem_text for x in [\n        "参数校准",\n        "参数依据",\n        "参数来源",\n    ]):\n        for sec in sections:\n            if (\n                "参数" in sec["title"]\n                or sec["section_type"]\n                in {\n                    "preprocessing",\n                    "sensitivity",\n                }\n            ):\n                add(sec)\n\n    if any(x in problem_text for x in [\n        "误差",\n        "验证",\n        "蒙特卡洛",\n        "置信",\n    ]):\n        for sec in sections:\n            if sec["section_type"] in {\n                "validation",\n                "sensitivity",\n                "error_analysis",\n            }:\n                add(sec)\n\n    # 4. 全文结构型问题：根据关键词相关性选正文\n    if (\n        not selected\n        or "全文结构" in sources\n    ):\n        keys = issue_keywords(\n            issue\n        )\n\n        scored = []\n\n        for sec in sections:\n            if sec["section_type"] in {\n                "front_matter",\n                "reference",\n                "appendix",\n            }:\n                continue\n\n            hay = (\n                sec["title"]\n                + " "\n                + sec["content"][:3000]\n            ).lower()\n\n            score = sum(\n                1\n                for k in keys\n                if k in hay\n            )\n\n            if score:\n                scored.append(\n                    (score, sec)\n                )\n\n        scored.sort(\n            key=lambda x: -x[0]\n        )\n\n        for _, sec in scored[:8]:\n            add(sec)\n\n    # 5. 不超过合理上下文\n    return selected[:10]\n\n\ndef build_clean_outline(\n    sections,\n):\n    lines = []\n\n    for sec in sections:\n        lines.append(\n            f"{sec[\'section_index\']:02d}. "\n            f"[{sec[\'section_type\']}] "\n            f"{sec[\'title\']}"\n        )\n\n    return "\\n".join(lines)\n\n\ndef build_verify_prompt(\n    issue,\n    sections,\n    outline,\n):\n    scope = choose_complete_scope(\n        issue,\n        sections,\n    )\n\n    blocks = [\n        "===== 完整论文目录 =====",\n        outline,\n        "",\n        "===== 候选问题 =====",\n        json.dumps(\n            issue,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        "",\n        "===== 核验范围说明 =====",\n        (\n            "下面每个标记为【完整章节正文】的内容，"\n            "都是 MinerU 解析后该标题下的完整正文，"\n            "不是摘要预览。"\n        ),\n    ]\n\n    total = 0\n\n    for sec in scope:\n        block = (\n            f"\\n\\n【完整章节正文】"\n            f"{sec[\'title\']}\\n"\n            f"{sec[\'content\'].strip()}\\n"\n            f"【本章节结束】"\n        )\n\n        if (\n            total + len(block)\n            > VERIFY_MAX_CONTEXT\n        ):\n            remaining = (\n                VERIFY_MAX_CONTEXT\n                - total\n            )\n\n            if remaining > 1000:\n                blocks.append(\n                    block[:remaining]\n                )\n                blocks.append(\n                    "\\n【注意：本章节因总上下文限制未完整提供】"\n                )\n            break\n\n        blocks.append(block)\n        total += len(block)\n\n    if not scope:\n        blocks.append(\n            "\\n未找到足够的目标章节。"\n            "此时应优先判 unsupported，"\n            "不得猜测。"\n        )\n\n    return "\\n".join(blocks)\n\n\ndef sanitize_evidence(text):\n    """\n    最后一层机械保护：\n    移除显式 char-count 论据。\n    """\n    if not text:\n        return ""\n\n    parts = re.split(\n        r"[；;\\n]",\n        text,\n    )\n\n    kept = []\n\n    for part in parts:\n        if re.search(\n            r"\\d+\\s*(?:chars?|字符)",\n            part,\n            flags=re.I,\n        ):\n            continue\n\n        if any(x in part for x in [\n            "篇幅较短",\n            "篇幅明显偏短",\n            "字数较少",\n        ]):\n            continue\n\n        if part.strip():\n            kept.append(\n                part.strip()\n            )\n\n    return "；".join(kept)\n\n\ndef verify_one_v24(\n    client,\n    issue,\n    sections,\n    outline,\n):\n    prompt = build_verify_prompt(\n        issue=issue,\n        sections=sections,\n        outline=outline,\n    )\n\n    data = v21.reliable_call_json(\n        client=client,\n        model=VERIFY_MODEL,\n        system_prompt=\n            VERIFY_SYSTEM_V24,\n        user_prompt=prompt,\n        max_tokens=4500,\n        retries=3,\n    )\n\n    verdict = data.get(\n        "verdict",\n        "unsupported",\n    )\n\n    if verdict not in {\n        "supported",\n        "refined",\n        "unsupported",\n    }:\n        verdict = "unsupported"\n\n    try:\n        confidence = float(\n            data.get(\n                "confidence",\n                0,\n            )\n        )\n    except Exception:\n        confidence = 0.0\n\n    accepted = (\n        verdict in {\n            "supported",\n            "refined",\n        }\n        and confidence >= 0.70\n        and bool(\n            data.get(\n                "scope_sufficient",\n                False,\n            )\n        )\n    )\n\n    result = dict(issue)\n\n    result[\n        "evidence_verification"\n    ] = {\n        "verdict": verdict,\n        "confidence": confidence,\n        "issue_kind":\n            data.get(\n                "issue_kind",\n                "",\n            ),\n        "evidence_basis":\n            data.get(\n                "evidence_basis",\n                "",\n            ),\n        "scope_sufficient":\n            bool(\n                data.get(\n                    "scope_sufficient",\n                    False,\n                )\n            ),\n        "verification_note":\n            data.get(\n                "verification_note",\n                "",\n            ),\n        "evidence_sections":\n            data.get(\n                "evidence_sections",\n                [],\n            ),\n    }\n\n    if accepted:\n        result["problem"] = (\n            data.get(\n                "problem"\n            )\n            or result.get(\n                "problem",\n                "",\n            )\n        )\n\n        importance = data.get(\n            "importance",\n            result.get(\n                "importance",\n                "medium",\n            ),\n        )\n\n        if importance not in {\n            "high",\n            "medium",\n            "low",\n        }:\n            importance = "medium"\n\n        result[\n            "importance"\n        ] = importance\n\n        result["reason"] = (\n            data.get(\n                "reason"\n            )\n            or result.get(\n                "reason",\n                "",\n            )\n        )\n\n        result[\n            "evidence_from_draft"\n        ] = sanitize_evidence(\n            data.get(\n                "evidence_from_draft",\n                "",\n            )\n        )\n\n        result[\n            "source_sections"\n        ] = (\n            data.get(\n                "evidence_sections"\n            )\n            or result.get(\n                "source_sections",\n                [],\n            )\n        )\n\n        new_type = data.get(\n            "retrieval_section_type"\n        )\n\n        if new_type in (\n            v21.base.SECTION_TYPES\n        ):\n            result[\n                "retrieval_section_type"\n            ] = new_type\n\n        if data.get(\n            "retrieval_query"\n        ):\n            result[\n                "retrieval_query"\n            ] = data[\n                "retrieval_query"\n            ]\n\n        if data.get(\n            "suggested_direction"\n        ):\n            result[\n                "suggested_direction"\n            ] = data[\n                "suggested_direction"\n            ]\n\n    return accepted, result\n\n\ndef verify_all_v24(\n    client,\n    merged_data,\n    sections,\n    outline,\n):\n    print(\n        "\\n[3.5/5] 正在执行 V2.4 "\n        "Evidence Calibration..."\n    )\n\n    accepted = []\n    rejected = []\n\n    issues = merged_data.get(\n        "issues",\n        [],\n    )\n\n    for i, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        print(\n            f"\\n[{i}/{len(issues)}] "\n            f"{issue.get(\'problem\')}"\n        )\n\n        ok, result = verify_one_v24(\n            client=client,\n            issue=issue,\n            sections=sections,\n            outline=outline,\n        )\n\n        ev = result[\n            "evidence_verification"\n        ]\n\n        print(\n            f"verdict={ev[\'verdict\']}  "\n            f"confidence={ev[\'confidence\']}  "\n            f"importance={result.get(\'importance\')}"\n        )\n\n        if (\n            ev["verdict"]\n            == "refined"\n        ):\n            print(\n                "  refined →",\n                result["problem"],\n            )\n\n        if ok:\n            accepted.append(result)\n        else:\n            rejected.append(result)\n\n    # 排序：high -> medium -> low\n    rank = {\n        "high": 0,\n        "medium": 1,\n        "low": 2,\n    }\n\n    accepted.sort(\n        key=lambda x: rank.get(\n            x.get(\n                "importance",\n                "medium",\n            ),\n            1,\n        )\n    )\n\n    print()\n    print(\n        "V2.4 Evidence Calibration："\n        f"通过/收窄 {len(accepted)}，"\n        f"淘汰 {len(rejected)}。"\n    )\n\n    return {\n        "overall_assessment":\n            merged_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues": accepted,\n        "rejected_issues":\n            rejected,\n    }\n\n\nFINAL_REPORT_SYSTEM_V24 = r"""\n你是一名数学建模竞赛论文指导型评审助手。\n\n你收到的所有问题都已经经过 V2.4 Evidence Calibration：\n- supported：原问题被完整章节支持；\n- refined：原问题被收窄为更精确的问题。\n\n不得新增任何未经核验的问题。\n\n严格规则：\n\n1. 只使用 evidence_from_draft 中的核验证据。\n\n2. 禁止使用：\n   “XX字符”\n   “XX chars”\n   “篇幅短”\n   “字数少”\n   作为问题依据或修改理由。\n\n3. importance 已经重新评定：\n   high 优先写；\n   medium 次之；\n   low 只作为呈现优化。\n   不得把 low 的结论排版问题写成“严重硬伤”。\n\n4. 对 mathematical_error：\n   必须说明为什么会导致数学不可行、错误解或逻辑冲突。\n\n5. 对 model_completeness：\n   区分“概念已经用文字表达”和“数学形式仍不完整”。\n\n6. 对 structural RAG：\n   必须明确说：\n   “模型场景不同，仅参考其……结构/表达方式”。\n\n7. 没有强相关案例时宁可说明没有，不得硬套。\n\n8. 不给无 Rubric 的百分比分数或总分。\n\n输出完整四部分：\n\n# 一、总体评价\n\n# 二、最优先修改的问题\n\n每个问题包含：\n- 问题是什么\n- 为什么重要\n- 草稿中的核验证据\n- 优秀论文案例怎么处理\n- 从案例抽象什么\n- 作者如何修改\n\n# 三、分模块修改建议\n\n# 四、修改顺序\n\n最后必须给：\nP0：必须优先改\nP1：重点改\nP2：有时间再优化\n"""\n\n\ndef build_reference_text_v24(\n    data,\n):\n    blocks = []\n\n    for i, issue in enumerate(\n        data.get(\n            "issues",\n            [],\n        ),\n        start=1,\n    ):\n        ev = issue.get(\n            "evidence_verification",\n            {},\n        )\n\n        blocks.extend([\n            (\n                f"## Issue {i}: "\n                f"{issue.get(\'problem\')}"\n            ),\n            (\n                "importance: "\n                f"{issue.get(\'importance\')}"\n            ),\n            (\n                "verification: "\n                f"{ev.get(\'verdict\')}"\n            ),\n            (\n                "issue_kind: "\n                f"{ev.get(\'issue_kind\')}"\n            ),\n            (\n                "evidence_basis: "\n                f"{ev.get(\'evidence_basis\')}"\n            ),\n            (\n                "source_sections: "\n                + ", ".join(\n                    issue.get(\n                        "source_sections",\n                        [],\n                    )\n                )\n            ),\n            (\n                "reason: "\n                + issue.get(\n                    "reason",\n                    "",\n                )\n            ),\n            (\n                "evidence: "\n                + sanitize_evidence(\n                    issue.get(\n                        "evidence_from_draft",\n                        "",\n                    )\n                )\n            ),\n            (\n                "suggested_direction: "\n                + issue.get(\n                    "suggested_direction",\n                    "",\n                )\n            ),\n        ])\n\n        examples = (\n            issue.get(\n                "retrieval",\n                {}\n            ).get(\n                "examples",\n                [],\n            )\n        )\n\n        if not examples:\n            blocks.append(\n                "RAG examples: none"\n            )\n            continue\n\n        for j, ex in enumerate(\n            examples,\n            start=1,\n        ):\n            blocks.append(\n                f"\\nExample {j}\\n"\n                f"relevance: "\n                f"{ex.get(\'relevance\')}\\n"\n                f"rerank_score: "\n                f"{ex.get(\'score\')}\\n"\n                f"rerank_reason: "\n                f"{ex.get(\'reason\')}\\n"\n                f"paper_id: "\n                f"{ex.get(\'paper_id\')}\\n"\n                f"section_title: "\n                f"{ex.get(\'section_title\')}\\n"\n                f"section_type: "\n                f"{ex.get(\'section_type\')}\\n"\n                f"content:\\n"\n                f"{ex.get(\'content\')}\\n"\n            )\n\n    return "\\n".join(\n        blocks\n    )\n\n\ndef report_complete(text):\n    return (\n        bool(text)\n        and all(\n            marker in text\n            for marker in [\n                "一、总体评价",\n                "二、最优先修改的问题",\n                "三、分模块修改建议",\n                "四、修改顺序",\n                "P0",\n                "P1",\n                "P2",\n            ]\n        )\n    )\n\n\ndef call_report(\n    client,\n    messages,\n    max_tokens,\n):\n    response = (\n        client.chat.completions.create(\n            model=REPORT_MODEL,\n            messages=messages,\n            temperature=0.2,\n            max_tokens=max_tokens,\n            extra_body={\n                "thinking": {\n                    "type":\n                        "disabled"\n                }\n            },\n        )\n    )\n\n    choice = response.choices[0]\n\n    return (\n        (\n            choice.message.content\n            or ""\n        ).strip(),\n        getattr(\n            choice,\n            "finish_reason",\n            None,\n        ),\n    )\n\n\ndef generate_report_v24(\n    client,\n    clean_outline,\n    enriched_data,\n):\n    print(\n        f"\\n[5/5] 正在生成 V2.4 最终报告..."\n    )\n\n    reference_text = (\n        build_reference_text_v24(\n            enriched_data\n        )\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{clean_outline}\\n\\n"\n        "===== 已核验问题 + "\n        "rerank优秀论文案例 =====\\n"\n        f"{reference_text}"\n    )\n\n    messages = [\n        {\n            "role": "system",\n            "content":\n                FINAL_REPORT_SYSTEM_V24,\n        },\n        {\n            "role": "user",\n            "content":\n                user_prompt,\n        },\n    ]\n\n    content, finish_reason = (\n        call_report(\n            client,\n            messages,\n            REPORT_MAX_TOKENS,\n        )\n    )\n\n    if not content:\n        raise RuntimeError(\n            "V2.4 最终报告为空"\n        )\n\n    full = content\n\n    n = 0\n\n    while (\n        finish_reason == "length"\n        and n < MAX_CONTINUATIONS\n    ):\n        n += 1\n\n        messages.append({\n            "role": "assistant",\n            "content": full,\n        })\n        messages.append({\n            "role": "user",\n            "content":\n                "从截断处继续，不重复，"\n                "最终完成四个一级部分和P0/P1/P2。",\n        })\n\n        cont, finish_reason = (\n            call_report(\n                client,\n                messages,\n                CONTINUE_MAX_TOKENS,\n            )\n        )\n\n        if not cont:\n            break\n\n        full = (\n            full.rstrip()\n            + "\\n\\n"\n            + cont.lstrip()\n        )\n\n    if re.search(\n        r"\\d+\\s*(?:chars?|字符)",\n        full,\n        flags=re.I,\n    ):\n        print(\n            "警告：最终报告仍出现字符数表达，"\n            "请把该日志发给我继续处理。"\n        )\n\n    if not report_complete(\n        full\n    ):\n        print(\n            "警告：V2.4 报告完整性检查未通过。"\n        )\n\n    return full.strip()\n\n\ndef main():\n    api_key = (\n        v21.base.require_api_key()\n    )\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python review_agent_deepseek_v2_4.py "\n            "\\"你的论文.md\\""\n        )\n        sys.exit(1)\n\n    paper_path, text = (\n        v21.base.read_paper(\n            sys.argv[1]\n        )\n    )\n\n    print("=" * 72)\n    print(\n        "数学建模整篇论文评审 Agent V2.4"\n    )\n    print("=" * 72)\n    print(\n        "论文：",\n        paper_path,\n    )\n    print(\n        "Evidence模型：",\n        VERIFY_MODEL,\n    )\n    print(\n        "报告模型：",\n        REPORT_MODEL,\n    )\n\n    work_dir = (\n        v21.make_work_dir(\n            paper_path,\n            text,\n        )\n    )\n\n    sections = (\n        v21.split_sections_v21(\n            text\n        )\n    )\n\n    clean_outline = (\n        build_clean_outline(\n            sections\n        )\n    )\n\n    client = (\n        v21.base.make_client(\n            api_key\n        )\n    )\n\n    # --------------------------\n    # 直接复用 04 merged\n    # --------------------------\n    merged_path = (\n        work_dir\n        / "04_merged_issues.json"\n    )\n\n    merged_data = load_json(\n        merged_path\n    )\n\n    if merged_data is None:\n        print(\n            "没有发现 04_merged_issues.json，"\n            "先执行 V2.1/V2.3 生成前置缓存。"\n        )\n        sys.exit(1)\n\n    print(\n        "[1~3/5] 检测到前置评审与合并缓存，直接复用。"\n    )\n\n    # --------------------------\n    # V2.4 Evidence\n    # --------------------------\n    verified_path = (\n        work_dir\n        / "04b_verified_issues_v24.json"\n    )\n\n    verified_data = load_json(\n        verified_path\n    )\n\n    if verified_data is None:\n        verified_data = (\n            verify_all_v24(\n                client=client,\n                merged_data=\n                    merged_data,\n                sections=sections,\n                outline=\n                    clean_outline,\n            )\n        )\n\n        save_json(\n            verified_path,\n            verified_data,\n        )\n    else:\n        print(\n            "[3.5/5] 检测到 V2.4 Evidence 缓存，直接复用。"\n        )\n\n    if not verified_data.get(\n        "issues"\n    ):\n        print(\n            "没有问题通过/收窄后保留，停止生成报告。"\n        )\n        return\n\n    # --------------------------\n    # 复用 V2.3 RAG rerank 逻辑\n    # --------------------------\n    retrieval_path = (\n        work_dir\n        / "05_retrieval_v24.json"\n    )\n\n    enriched_data = load_json(\n        retrieval_path\n    )\n\n    if enriched_data is None:\n        collection, embedding_model = (\n            v21.base.initialize_retrieval()\n        )\n\n        enriched_data = (\n            v23.retrieve_v23(\n                client=client,\n                verified_data=\n                    verified_data,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        save_json(\n            retrieval_path,\n            enriched_data,\n        )\n    else:\n        print(\n            "[4/5] 检测到 V2.4 RAG 缓存，直接复用。"\n        )\n\n    # --------------------------\n    # final\n    # --------------------------\n    report_path = (\n        work_dir\n        / "review_report_v24.md"\n    )\n\n    if (\n        report_path.exists()\n        and report_complete(\n            report_path.read_text(\n                encoding="utf-8",\n                errors="ignore",\n            )\n        )\n    ):\n        report = report_path.read_text(\n            encoding="utf-8",\n        )\n        print(\n            "[5/5] 检测到完整 V2.4 报告，直接复用。"\n        )\n    else:\n        report = (\n            generate_report_v24(\n                client=client,\n                clean_outline=\n                    clean_outline,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        report_path.write_text(\n            report,\n            encoding="utf-8",\n        )\n\n    print()\n    print("=" * 72)\n    print(\n        "V2.4 最终评审报告"\n    )\n    print("=" * 72)\n    print()\n    print(report)\n\n    print()\n    print("=" * 72)\n    print(\n        "Evidence：",\n        verified_path,\n    )\n    print(\n        "RAG：",\n        retrieval_path,\n    )\n    print(\n        "报告：",\n        report_path,\n    )\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2_4 = _load_embedded_module('review_agent_deepseek_v2_4', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2_4)

# ===== Embedded module: review_agent_deepseek_v2_5 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2_5 = '"""\nreview_agent_deepseek_v2_5.py\n\nV2.5：整篇数学建模论文评审 Agent 收口版\n\n核心目标：\n1. 复用 V2.4 Evidence Calibration。\n2. 在 Evidence 后增加 Post-Evidence Consolidation：\n   - 合并高度重叠 issue\n   - 不新增未经核验的问题\n   - 保留原证据\n   - 生成稳定 issue_id\n   - 最终保留约 5~7 个互不重叠问题\n3. importance 锁定：\n   high   -> P0\n   medium -> P1\n   low    -> P2\n4. RAG 在 consolidation 之后重新执行。\n5. 最终报告不能新增未经核验/合并的问题。\n6. “四、修改顺序”由程序生成，不交给大模型自由决定，保证优先级一致。\n"""\n\nimport json\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n\nimport review_agent_deepseek_v2_1 as v21\nimport review_agent_deepseek_v2_3 as v23\nimport review_agent_deepseek_v2_4 as v24\n\n\nCONSOLIDATE_MODEL = os.getenv(\n    "DEEPSEEK_CONSOLIDATE_MODEL",\n    "deepseek-v4-pro",\n)\n\nREPORT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_MODEL",\n    "deepseek-v4-pro",\n)\n\nMAX_CONSOLIDATED_ISSUES = 7\n\nREPORT_MAX_TOKENS = 28000\nCONTINUE_MAX_TOKENS = 16000\nMAX_CONTINUATIONS = 2\n\n\n# ============================================================\n# 通用 IO\n# ============================================================\n\ndef save_json(path, data):\n    path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path, default=None):\n    if not path.exists():\n        return default\n\n    try:\n        return json.loads(\n            path.read_text(\n                encoding="utf-8"\n            )\n        )\n    except Exception:\n        return default\n\n\n# ============================================================\n# 1. Post-Evidence Consolidation\n# ============================================================\n\nCONSOLIDATE_SYSTEM = r"""\n你是数学建模竞赛论文评审系统中的“Evidence 后问题合并器”。\n\n输入中的所有问题已经经过 Evidence Calibration：\n- supported：原问题被完整正文支持\n- refined：原问题已经被收窄成证据可支持的版本\n\n你的任务不是继续找问题，而是把这些已核验问题整理成一组：\n“互不重复、边界清晰、优先级稳定”的最终 issue。\n\n====================\n一、绝对禁止\n====================\n\n1. 不得新增输入中不存在的新问题。\n2. 不得把输入 evidence 没支持的推测写进去。\n3. 不得因为想凑数量而拆分问题。\n4. 不得因为想减少数量而把本质不同的问题硬合并。\n5. 不得修改问题的数学事实。\n6. 不得使用“字数少 / 篇幅短 / xx chars”作为证据。\n\n====================\n二、什么时候合并\n====================\n\n应合并：\n- 同一参数问题在不同 issue 中重复出现；\n- 一个“大而泛”的 issue 与一个更具体 issue 高度重叠；\n- 同一验证环节的“统计量不足”和“误差度量不足”高度相关；\n- 同一 SAA 问题分别被写成“模型建立细节不足”“SAA形式不完整”“求解可复现性不足”。\n\n合并后要保留各子问题的具体证据。\n\n不应合并：\n- 数学不可行性问题 与 参数依据问题；\n- 模型形式化问题 与 结果呈现问题；\n- 模型正确性问题 与 结论排版问题。\n\n====================\n三、importance\n====================\n\nimportance 必须从输入问题中综合判断，不得随意升级。\n\nhigh：\n- 数学不可行性 / 逻辑错误 / 必要约束缺失；\n- 会直接影响模型正确性。\n\nmedium：\n- 参数依据不足；\n- 形式化不完整；\n- 可复现性不足；\n- 验证定量不足；\n- 结果定量解释不足。\n\nlow：\n- 结论组织；\n- 表格化；\n- 逻辑串联；\n- 一般呈现优化。\n\n如果一个合并 issue 同时含 high 和 medium 子项，\n只有当 high 子项是该 issue 的核心时才保留 high。\n\n====================\n四、期望数量\n====================\n\n如果输入问题 >= 7 个：\n优先整理为 5~7 个独立问题。\n\n如果输入本身少于 5 个：\n不要为了达到 5 个而拆分。\n\n====================\n五、输出\n====================\n\n只输出合法 JSON：\n\n{\n  "overall_assessment": "简短总体判断",\n  "issues": [\n    {\n      "problem": "最终独立问题名称",\n      "importance": "high|medium|low",\n      "issue_kind": "mathematical_error|model_completeness|parameter_basis|reproducibility|validation_gap|presentation",\n      "reason": "合并后原因",\n      "evidence_from_draft": "由已核验证据合并而来",\n      "source_sections": ["章节"],\n      "retrieval_section_type": "合法section_type",\n      "retrieval_query": "针对最终问题的知识库检索语句",\n      "suggested_direction": "修改方向",\n      "merged_from": ["原问题名称1", "原问题名称2"]\n    }\n  ]\n}\n"""\n\n\ndef importance_rank(value):\n    return {\n        "high": 0,\n        "medium": 1,\n        "low": 2,\n    }.get(\n        value,\n        1,\n    )\n\n\ndef issue_kind_default(issue):\n    ev = issue.get(\n        "evidence_verification",\n        {},\n    )\n\n    return (\n        ev.get("issue_kind")\n        or "model_completeness"\n    )\n\n\ndef compact_verified_for_prompt(\n    verified_data,\n):\n    items = []\n\n    for idx, issue in enumerate(\n        verified_data.get(\n            "issues",\n            [],\n        ),\n        start=1,\n    ):\n        ev = issue.get(\n            "evidence_verification",\n            {},\n        )\n\n        items.append({\n            "source_index": idx,\n            "problem":\n                issue.get(\n                    "problem",\n                    "",\n                ),\n            "importance":\n                issue.get(\n                    "importance",\n                    "medium",\n                ),\n            "issue_kind":\n                issue_kind_default(\n                    issue\n                ),\n            "reason":\n                issue.get(\n                    "reason",\n                    "",\n                ),\n            "evidence_from_draft":\n                v24.sanitize_evidence(\n                    issue.get(\n                        "evidence_from_draft",\n                        "",\n                    )\n                ),\n            "source_sections":\n                issue.get(\n                    "source_sections",\n                    [],\n                ),\n            "retrieval_section_type":\n                issue.get(\n                    "retrieval_section_type",\n                    "model_detail",\n                ),\n            "retrieval_query":\n                issue.get(\n                    "retrieval_query",\n                    "",\n                ),\n            "suggested_direction":\n                issue.get(\n                    "suggested_direction",\n                    "",\n                ),\n            "verification_verdict":\n                ev.get(\n                    "verdict",\n                    "",\n                ),\n            "evidence_basis":\n                ev.get(\n                    "evidence_basis",\n                    "",\n                ),\n        })\n\n    return items\n\n\ndef normalize_consolidated_issue(\n    issue,\n):\n    if not isinstance(\n        issue,\n        dict,\n    ):\n        return None\n\n    importance = issue.get(\n        "importance",\n        "medium",\n    )\n\n    if importance not in {\n        "high",\n        "medium",\n        "low",\n    }:\n        importance = "medium"\n\n    issue_kind = issue.get(\n        "issue_kind",\n        "model_completeness",\n    )\n\n    if issue_kind not in {\n        "mathematical_error",\n        "model_completeness",\n        "parameter_basis",\n        "reproducibility",\n        "validation_gap",\n        "presentation",\n    }:\n        issue_kind = (\n            "model_completeness"\n        )\n\n    section_type = issue.get(\n        "retrieval_section_type",\n        "model_detail",\n    )\n\n    if section_type not in (\n        v21.base.SECTION_TYPES\n    ):\n        section_type = (\n            "model_detail"\n        )\n\n    evidence = (\n        v24.sanitize_evidence(\n            issue.get(\n                "evidence_from_draft",\n                "",\n            )\n        )\n    )\n\n    if not issue.get(\n        "problem"\n    ):\n        return None\n\n    return {\n        "problem":\n            issue.get(\n                "problem",\n                "",\n            ).strip(),\n        "importance":\n            importance,\n        "issue_kind":\n            issue_kind,\n        "reason":\n            issue.get(\n                "reason",\n                "",\n            ).strip(),\n        "evidence_from_draft":\n            evidence,\n        "source_sections":\n            list(dict.fromkeys(\n                issue.get(\n                    "source_sections",\n                    [],\n                )\n            )),\n        "retrieval_section_type":\n            section_type,\n        "retrieval_query":\n            issue.get(\n                "retrieval_query",\n                issue.get(\n                    "problem",\n                    "",\n                ),\n            ).strip(),\n        "suggested_direction":\n            issue.get(\n                "suggested_direction",\n                "",\n            ).strip(),\n        "merged_from":\n            list(dict.fromkeys(\n                issue.get(\n                    "merged_from",\n                    [],\n                )\n            )),\n    }\n\n\ndef assign_issue_ids(\n    issues,\n):\n    issues = sorted(\n        issues,\n        key=lambda x: (\n            importance_rank(\n                x.get(\n                    "importance",\n                    "medium",\n                )\n            ),\n            x.get(\n                "problem",\n                "",\n            ),\n        )\n    )\n\n    result = []\n\n    for idx, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        x = dict(issue)\n\n        x["issue_id"] = (\n            f"I{idx:02d}"\n        )\n\n        x["priority"] = {\n            "high": "P0",\n            "medium": "P1",\n            "low": "P2",\n        }[\n            x["importance"]\n        ]\n\n        # Consolidated issue 已经来自\n        # V2.4 verified/refined issues\n        x[\n            "evidence_verification"\n        ] = {\n            "verdict":\n                "post_evidence_consolidated",\n            "confidence": 1.0,\n            "issue_kind":\n                x["issue_kind"],\n            "evidence_basis":\n                "verified_inputs_only",\n            "scope_sufficient":\n                True,\n            "verification_note":\n                "本问题仅由V2.4已核验问题合并而来，"\n                "未新增未经证实的问题。",\n            "evidence_sections":\n                x[\n                    "source_sections"\n                ],\n        }\n\n        result.append(x)\n\n    return result\n\n\ndef consolidate_verified_issues(\n    client,\n    verified_data,\n):\n    print(\n        "\\n[3.7/5] 正在执行 "\n        "Post-Evidence Consolidation..."\n    )\n\n    source_issues = (\n        compact_verified_for_prompt(\n            verified_data\n        )\n    )\n\n    if not source_issues:\n        return {\n            "overall_assessment":\n                verified_data.get(\n                    "overall_assessment",\n                    "",\n                ),\n            "issues": [],\n            "source_issue_count": 0,\n        }\n\n    print(\n        f"Evidence 后问题数："\n        f"{len(source_issues)}"\n    )\n\n    user_prompt = (\n        "下面是同一篇论文经过 Evidence "\n        "Calibration 后的全部问题。\\n"\n        "请只合并重复/高度重叠项，"\n        "不要增加新问题。\\n\\n"\n        + json.dumps(\n            source_issues,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    data = (\n        v21.reliable_call_json(\n            client=client,\n            model=CONSOLIDATE_MODEL,\n            system_prompt=\n                CONSOLIDATE_SYSTEM,\n            user_prompt=user_prompt,\n            max_tokens=8500,\n            retries=3,\n        )\n    )\n\n    normalized = []\n\n    for raw in data.get(\n        "issues",\n        [],\n    ):\n        x = (\n            normalize_consolidated_issue(\n                raw\n            )\n        )\n\n        if x:\n            normalized.append(x)\n\n    # 如果模型仍给太多，执行第二轮压缩\n    if (\n        len(normalized)\n        > MAX_CONSOLIDATED_ISSUES\n    ):\n        print(\n            f"第一轮仍有 {len(normalized)} 个，"\n            f"执行最终压缩到 "\n            f"{MAX_CONSOLIDATED_ISSUES} 个以内..."\n        )\n\n        user_prompt2 = (\n            "下面已经是第一轮合并后的问题，"\n            "但仍然过多。\\n"\n            f"请继续只合并高度重叠项，"\n            f"最终不超过 "\n            f"{MAX_CONSOLIDATED_ISSUES} 个。"\n            "不得删除唯一的 high 数学错误，"\n            "不得新增问题。\\n\\n"\n            + json.dumps(\n                normalized,\n                ensure_ascii=False,\n                indent=2,\n            )\n        )\n\n        data2 = (\n            v21.reliable_call_json(\n                client=client,\n                model=CONSOLIDATE_MODEL,\n                system_prompt=\n                    CONSOLIDATE_SYSTEM,\n                user_prompt=\n                    user_prompt2,\n                max_tokens=8000,\n                retries=3,\n            )\n        )\n\n        normalized2 = []\n\n        for raw in data2.get(\n            "issues",\n            [],\n        ):\n            x = (\n                normalize_consolidated_issue(\n                    raw\n                )\n            )\n\n            if x:\n                normalized2.append(x)\n\n        if normalized2:\n            normalized = (\n                normalized2\n            )\n\n    # 最后保险：若仍超过上限，\n    # 按 importance 排序截取。\n    normalized.sort(\n        key=lambda x:\n            importance_rank(\n                x["importance"]\n            )\n    )\n\n    normalized = normalized[\n        :MAX_CONSOLIDATED_ISSUES\n    ]\n\n    normalized = assign_issue_ids(\n        normalized\n    )\n\n    print(\n        f"Post-Evidence Consolidation："\n        f"{len(source_issues)} → "\n        f"{len(normalized)}"\n    )\n\n    for issue in normalized:\n        print(\n            f"  {issue[\'issue_id\']} "\n            f"[{issue[\'importance\']}/{issue[\'priority\']}] "\n            f"{issue[\'problem\']}"\n        )\n\n    return {\n        "overall_assessment":\n            data.get(\n                "overall_assessment",\n                verified_data.get(\n                    "overall_assessment",\n                    "",\n                ),\n            ),\n        "source_issue_count":\n            len(source_issues),\n        "issues":\n            normalized,\n    }\n\n\n# ============================================================\n# 2. RAG 复用 V2.3\n# ============================================================\n\ndef retrieve_after_consolidation(\n    client,\n    consolidated_data,\n    embedding_model,\n    collection,\n):\n    print(\n        "\\n[4/5] 正在为 consolidation "\n        "后的独立问题执行 RAG + rerank..."\n    )\n\n    # v23.retrieve_v23 会保留 issue_id 等字段\n    return v23.retrieve_v23(\n        client=client,\n        verified_data=\n            consolidated_data,\n        embedding_model=\n            embedding_model,\n        collection=collection,\n    )\n\n\n# ============================================================\n# 3. 最终报告\n# ============================================================\n\nFINAL_REPORT_SYSTEM_V25 = r"""\n你是一名数学建模竞赛论文指导型评审助手。\n\n你收到的是：\n【经过 Evidence Calibration + Post-Evidence Consolidation 后的最终问题】\n\n这些 issue 已经是唯一允许写进报告的问题集合。\n\n绝对规则：\n\n1. 不得新增任何新的问题。\n2. 不得在“分模块修改建议”中偷偷增加新的批评点。\n3. 所有批评、修改建议必须能对应到一个 issue_id。\n4. 可以解释、展开已有 issue，但不能发明新 issue。\n5. strong 案例可以重点参考；\n   structural 案例必须明确说“模型场景不同，仅参考其结构/表达方式”。\n6. 禁止使用“字数少/篇幅短/xx chars”作为问题证据。\n7. 不给无 Rubric 的百分比总分。\n8. importance 已锁定，不得升级/降级。\n\n报告只生成前三部分：\n\n# 一、总体评价\n\n总体评价必须围绕现有 issue。\n可以说明论文已有优点，但不得凭空增加新的缺陷。\n\n# 二、最终独立问题\n\n必须按 issue_id 顺序逐项写。\n\n每项格式：\n\n## Ixx：问题名（importance / priority）\n\n### 问题是什么\n### 为什么重要\n### 草稿中的核验证据\n### 优秀论文案例怎么处理\n### 从案例抽象什么\n### 作者如何修改\n\n# 三、分模块修改建议\n\n这一部分只允许重新组织已有 issue 的建议。\n\n每个模块都必须显式写：\n“对应 issue：Ixx, Iyy”\n\n如果某模块没有对应 issue，就不要为了完整性硬写建议。\n\n不要生成“四、修改顺序”。\n第四部分由程序根据 importance 自动生成。\n"""\n\n\ndef build_reference_text_v25(\n    enriched_data,\n):\n    blocks = []\n\n    for issue in enriched_data.get(\n        "issues",\n        [],\n    ):\n        blocks.extend([\n            (\n                f"## {issue[\'issue_id\']} "\n                f"{issue[\'problem\']}"\n            ),\n            (\n                "importance: "\n                f"{issue[\'importance\']}"\n            ),\n            (\n                "priority: "\n                f"{issue[\'priority\']}"\n            ),\n            (\n                "issue_kind: "\n                f"{issue.get(\'issue_kind\')}"\n            ),\n            (\n                "source_sections: "\n                + ", ".join(\n                    issue.get(\n                        "source_sections",\n                        [],\n                    )\n                )\n            ),\n            (\n                "reason: "\n                + issue.get(\n                    "reason",\n                    "",\n                )\n            ),\n            (\n                "evidence: "\n                + v24.sanitize_evidence(\n                    issue.get(\n                        "evidence_from_draft",\n                        "",\n                    )\n                )\n            ),\n            (\n                "suggested_direction: "\n                + issue.get(\n                    "suggested_direction",\n                    "",\n                )\n            ),\n            (\n                "merged_from: "\n                + " | ".join(\n                    issue.get(\n                        "merged_from",\n                        [],\n                    )\n                )\n            ),\n        ])\n\n        examples = (\n            issue.get(\n                "retrieval",\n                {}\n            ).get(\n                "examples",\n                [],\n            )\n        )\n\n        if not examples:\n            blocks.append(\n                "RAG examples: none"\n            )\n            continue\n\n        for j, ex in enumerate(\n            examples,\n            start=1,\n        ):\n            blocks.append(\n                f"\\nExample {j}\\n"\n                f"relevance: "\n                f"{ex.get(\'relevance\')}\\n"\n                f"rerank_score: "\n                f"{ex.get(\'score\')}\\n"\n                f"rerank_reason: "\n                f"{ex.get(\'reason\')}\\n"\n                f"paper_id: "\n                f"{ex.get(\'paper_id\')}\\n"\n                f"section_title: "\n                f"{ex.get(\'section_title\')}\\n"\n                f"section_type: "\n                f"{ex.get(\'section_type\')}\\n"\n                f"content:\\n"\n                f"{ex.get(\'content\')}\\n"\n            )\n\n    return "\\n".join(\n        blocks\n    )\n\n\ndef call_report(\n    client,\n    messages,\n    max_tokens,\n):\n    response = (\n        client.chat.completions.create(\n            model=REPORT_MODEL,\n            messages=messages,\n            temperature=0.2,\n            max_tokens=max_tokens,\n            extra_body={\n                "thinking": {\n                    "type": "disabled"\n                }\n            },\n        )\n    )\n\n    choice = response.choices[0]\n\n    return (\n        (\n            choice.message.content\n            or ""\n        ).strip(),\n        getattr(\n            choice,\n            "finish_reason",\n            None,\n        ),\n    )\n\n\ndef generate_priority_section(\n    issues,\n):\n    groups = {\n        "P0": [],\n        "P1": [],\n        "P2": [],\n    }\n\n    for issue in issues:\n        groups[\n            issue["priority"]\n        ].append(issue)\n\n    lines = [\n        "# 四、修改顺序",\n        "",\n    ]\n\n    labels = {\n        "P0":\n            "P0：必须优先改",\n        "P1":\n            "P1：重点改",\n        "P2":\n            "P2：有时间再优化",\n    }\n\n    for priority in [\n        "P0",\n        "P1",\n        "P2",\n    ]:\n        lines.append(\n            f"## {labels[priority]}"\n        )\n        lines.append("")\n\n        items = groups[\n            priority\n        ]\n\n        if not items:\n            lines.append(\n                "本轮没有该优先级的问题。"\n            )\n            lines.append("")\n            continue\n\n        for idx, issue in enumerate(\n            items,\n            start=1,\n        ):\n            sections = "、".join(\n                issue.get(\n                    "source_sections",\n                    [],\n                )\n            )\n\n            line = (\n                f"{idx}. **{issue[\'issue_id\']} "\n                f"{issue[\'problem\']}**"\n            )\n\n            if sections:\n                line += (\n                    f"。修改位置：{sections}"\n                )\n\n            line += "。"\n\n            lines.append(line)\n\n        lines.append("")\n\n    return "\\n".join(\n        lines\n    ).strip()\n\n\ndef report_complete(\n    text,\n    issues,\n):\n    if not text:\n        return False\n\n    required = [\n        "一、总体评价",\n        "二、最终独立问题",\n        "三、分模块修改建议",\n        "四、修改顺序",\n        "P0",\n        "P1",\n        "P2",\n    ]\n\n    if not all(\n        x in text\n        for x in required\n    ):\n        return False\n\n    # 每个 issue_id 必须出现在报告\n    return all(\n        issue["issue_id"] in text\n        for issue in issues\n    )\n\n\ndef generate_report_v25(\n    client,\n    outline,\n    enriched_data,\n):\n    print(\n        "\\n[5/5] 正在生成 V2.5 最终报告..."\n    )\n\n    reference_text = (\n        build_reference_text_v25(\n            enriched_data\n        )\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline}\\n\\n"\n        "===== 最终允许使用的问题 =====\\n"\n        f"{reference_text}\\n\\n"\n        "再次强调：只能写这些 issue，"\n        "所有修改建议必须显式对应 issue_id。"\n    )\n\n    messages = [\n        {\n            "role": "system",\n            "content":\n                FINAL_REPORT_SYSTEM_V25,\n        },\n        {\n            "role": "user",\n            "content":\n                user_prompt,\n        },\n    ]\n\n    content, finish_reason = (\n        call_report(\n            client=client,\n            messages=messages,\n            max_tokens=\n                REPORT_MAX_TOKENS,\n        )\n    )\n\n    if not content:\n        raise RuntimeError(\n            "V2.5 报告前三部分为空"\n        )\n\n    full = content\n\n    count = 0\n\n    while (\n        finish_reason == "length"\n        and count\n        < MAX_CONTINUATIONS\n    ):\n        count += 1\n\n        messages.append({\n            "role": "assistant",\n            "content": full,\n        })\n\n        messages.append({\n            "role": "user",\n            "content":\n                "从截断处继续，"\n                "不要重复。"\n                "只完成前三部分，"\n                "不得新增issue。",\n        })\n\n        cont, finish_reason = (\n            call_report(\n                client=client,\n                messages=messages,\n                max_tokens=\n                    CONTINUE_MAX_TOKENS,\n            )\n        )\n\n        if not cont:\n            break\n\n        full = (\n            full.rstrip()\n            + "\\n\\n"\n            + cont.lstrip()\n        )\n\n    # 防止模型误生成第四部分\n    marker = re.search(\n        r"(?m)^#\\s*四、",\n        full,\n    )\n\n    if marker:\n        full = (\n            full[\n                :marker.start()\n            ].rstrip()\n        )\n\n    priority_section = (\n        generate_priority_section(\n            enriched_data.get(\n                "issues",\n                [],\n            )\n        )\n    )\n\n    final_report = (\n        full.rstrip()\n        + "\\n\\n"\n        + priority_section\n        + "\\n"\n    )\n\n    if re.search(\n        r"\\d+\\s*(?:chars?|字符)",\n        final_report,\n        flags=re.I,\n    ):\n        print(\n            "警告：报告中仍出现字符数表达。"\n        )\n\n    return final_report.strip()\n\n\n# ============================================================\n# 4. 主程序\n# ============================================================\n\ndef main():\n    api_key = (\n        v21.base.require_api_key()\n    )\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python review_agent_deepseek_v2_5.py "\n            "\\"你的论文.md\\""\n        )\n        sys.exit(1)\n\n    paper_path, text = (\n        v21.base.read_paper(\n            sys.argv[1]\n        )\n    )\n\n    print("=" * 72)\n    print(\n        "数学建模整篇论文评审 Agent V2.5"\n    )\n    print("=" * 72)\n    print(\n        "论文：",\n        paper_path,\n    )\n    print(\n        "Consolidate模型：",\n        CONSOLIDATE_MODEL,\n    )\n    print(\n        "报告模型：",\n        REPORT_MODEL,\n    )\n\n    work_dir = (\n        v21.make_work_dir(\n            paper_path,\n            text,\n        )\n    )\n\n    sections = (\n        v21.split_sections_v21(\n            text\n        )\n    )\n\n    outline = (\n        v24.build_clean_outline(\n            sections\n        )\n    )\n\n    client = (\n        v21.base.make_client(\n            api_key\n        )\n    )\n\n    # --------------------------------------------------------\n    # V2.4 Evidence\n    # --------------------------------------------------------\n\n    verified_v24_path = (\n        work_dir\n        / "04b_verified_issues_v24.json"\n    )\n\n    verified_data = load_json(\n        verified_v24_path\n    )\n\n    if verified_data is None:\n        merged_path = (\n            work_dir\n            / "04_merged_issues.json"\n        )\n\n        merged_data = load_json(\n            merged_path\n        )\n\n        if merged_data is None:\n            print(\n                "没有发现前置 merged cache。"\n                "请先运行 V2.4。"\n            )\n            sys.exit(1)\n\n        verified_data = (\n            v24.verify_all_v24(\n                client=client,\n                merged_data=\n                    merged_data,\n                sections=sections,\n                outline=outline,\n            )\n        )\n\n        save_json(\n            verified_v24_path,\n            verified_data,\n        )\n    else:\n        print(\n            "[1~3.5/5] 检测到 V2.4 Evidence 缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Post-Evidence Consolidation\n    # --------------------------------------------------------\n\n    consolidated_path = (\n        work_dir\n        / "04c_consolidated_issues_v25.json"\n    )\n\n    consolidated_data = (\n        load_json(\n            consolidated_path\n        )\n    )\n\n    if consolidated_data is None:\n        consolidated_data = (\n            consolidate_verified_issues(\n                client=client,\n                verified_data=\n                    verified_data,\n            )\n        )\n\n        save_json(\n            consolidated_path,\n            consolidated_data,\n        )\n    else:\n        print(\n            "[3.7/5] 检测到 V2.5 consolidation 缓存，直接复用。"\n        )\n\n    if not consolidated_data.get(\n        "issues"\n    ):\n        print(\n            "Consolidation 后没有可保留问题，停止。"\n        )\n        return\n\n    # --------------------------------------------------------\n    # RAG\n    # --------------------------------------------------------\n\n    retrieval_path = (\n        work_dir\n        / "05_retrieval_v25.json"\n    )\n\n    enriched_data = (\n        load_json(\n            retrieval_path\n        )\n    )\n\n    if enriched_data is None:\n        collection, embedding_model = (\n            v21.base.initialize_retrieval()\n        )\n\n        enriched_data = (\n            retrieve_after_consolidation(\n                client=client,\n                consolidated_data=\n                    consolidated_data,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        save_json(\n            retrieval_path,\n            enriched_data,\n        )\n    else:\n        print(\n            "[4/5] 检测到 V2.5 RAG 缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Final report\n    # --------------------------------------------------------\n\n    report_path = (\n        work_dir\n        / "review_report_v25.md"\n    )\n\n    old_report = ""\n\n    if report_path.exists():\n        old_report = (\n            report_path.read_text(\n                encoding="utf-8",\n                errors="ignore",\n            )\n        )\n\n    if report_complete(\n        old_report,\n        enriched_data.get(\n            "issues",\n            [],\n        ),\n    ):\n        report = old_report\n        print(\n            "[5/5] 检测到完整 V2.5 报告，直接复用。"\n        )\n    else:\n        report = (\n            generate_report_v25(\n                client=client,\n                outline=outline,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        report_path.write_text(\n            report,\n            encoding="utf-8",\n        )\n\n    print()\n    print("=" * 72)\n    print(\n        "V2.5 最终评审报告"\n    )\n    print("=" * 72)\n    print()\n    print(report)\n\n    print()\n    print("=" * 72)\n    print(\n        "Consolidated：",\n        consolidated_path,\n    )\n    print(\n        "RAG：",\n        retrieval_path,\n    )\n    print(\n        "报告：",\n        report_path,\n    )\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2_5 = _load_embedded_module('review_agent_deepseek_v2_5', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2_5)

# ===== Embedded module: review_agent_deepseek_v2_5_1 =====
_SOURCE_REVIEW_AGENT_DEEPSEEK_V2_5_1 = '"""\nreview_agent_deepseek_v2_5_1.py\n\nV2.5.1：V2.5 收口版的事实正确性热修。\n\n不改前面的架构，只增加三道保险：\n\nA. Cross-section Fact Audit\n   - Consolidation 后，回看同一大章节及相关章节全文\n   - 防止“5.3没写”被误判成“整篇没写”\n   - 已在相邻章节出现的事实必须从批评中删除\n   - 例如：5.2 已写 N=6000，则不能再批评“SAA样本量未说明”\n\nB. Overlap Audit\n   - 对事实审计后的 issue 再做一次“子问题归属唯一化”\n   - 一个子问题只能归属于一个最终 issue\n   - 删除纯粹重复的 umbrella issue\n   - 目标通常为 4~6 个独立问题\n\nC. Final Claim Lock\n   - 最终报告关于“用户论文现状”的事实，只能来自：\n       evidence_from_draft / known_facts\n   - RAG 案例中的事实只能作为“优秀论文案例”，不能冒充用户论文事实\n   - 禁止自行定义符号\n   - 禁止编造示例数值\n   - 报告生成后用 Flash 再做一次 factual-claim audit；\n     若不通过，则自动让 Pro 修复一次\n\n输入：\n    python review_agent_deepseek_v2_5_1.py "paper.md"\n\n依赖：\n    review_agent_deepseek_v2_1.py\n    review_agent_deepseek_v2_3.py\n    review_agent_deepseek_v2_4.py\n    review_agent_deepseek_v2_5.py\n"""\n\nimport json\nimport os\nimport re\nimport sys\nfrom pathlib import Path\n\nimport review_agent_deepseek_v2_1 as v21\nimport review_agent_deepseek_v2_3 as v23\nimport review_agent_deepseek_v2_4 as v24\nimport review_agent_deepseek_v2_5 as v25\n\n\nFACT_AUDIT_MODEL = os.getenv(\n    "DEEPSEEK_FACT_AUDIT_MODEL",\n    "deepseek-v4-flash",\n)\n\nOVERLAP_MODEL = os.getenv(\n    "DEEPSEEK_OVERLAP_MODEL",\n    "deepseek-v4-pro",\n)\n\nREPORT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_MODEL",\n    "deepseek-v4-pro",\n)\n\nREPORT_AUDIT_MODEL = os.getenv(\n    "DEEPSEEK_REPORT_AUDIT_MODEL",\n    "deepseek-v4-flash",\n)\n\nMAX_FINAL_ISSUES = 6\n\nFACT_SCOPE_MAX_CHARS = 32000\n\nREPORT_MAX_TOKENS = 28000\nREPORT_REPAIR_MAX_TOKENS = 28000\n\n\n# ============================================================\n# IO\n# ============================================================\n\ndef save_json(path, data):\n    path.parent.mkdir(\n        parents=True,\n        exist_ok=True,\n    )\n    path.write_text(\n        json.dumps(\n            data,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        encoding="utf-8",\n    )\n\n\ndef load_json(path, default=None):\n    if not path.exists():\n        return default\n\n    try:\n        return json.loads(\n            path.read_text(\n                encoding="utf-8"\n            )\n        )\n    except Exception:\n        return default\n\n\n# ============================================================\n# 1. Cross-section Fact Audit\n# ============================================================\n\nFACT_AUDIT_SYSTEM = r"""\n你是数学建模论文评审系统中的“跨章节事实审计器”。\n\n输入包含：\n1. 一个已经经过 Evidence Calibration + 初步 Consolidation 的 issue；\n2. 与该 issue 有关的【完整章节正文】；\n3. 论文目录。\n\n你的任务不是新增问题，而是检查：\n“当前 issue 是否遗漏了相邻章节中已经写过的事实，从而把问题说重了？”\n\n====================\n核心原则\n====================\n\n1. 跨章节核验\n不能只看 issue 指向的一个小节。\n如果 5.3 没写样本量，但 5.2 已明确写 N=6000，\n则不能再说“SAA样本量未说明”。\n\n2. 区分“没有”与“写在别处”\n- 完全没有：可以保留批评\n- 已在其他相关章节写明：必须从批评中删除\n- 写了但不完整：收窄为真正缺失的部分\n\n3. 不新增新问题\n你只能：\n- keep：保持原 issue\n- corrected：删除错误子项/收窄问题\n- drop：如果整个问题已被跨章节证据反驳\n\n4. 不得自己定义符号\n如果正文没有明确给出某个符号的含义，\n不要自行猜测它是“基准票价”“成本参数”等。\n\n5. 不得编造数值\n不能用“N=1000”“提升4.7%”之类虚构数字。\n\n6. “建议怎么改”可以给数学结构或符号形式，\n但不能把建议中的内容写成论文已经存在的事实。\n\n7. importance 不得升级。\n本次审计主要修事实，不负责重新抬高严重性。\n\n====================\n输出 JSON\n====================\n\n{\n  "status": "keep|corrected|drop",\n  "problem": "审计后的问题名",\n  "reason": "审计后的原因",\n  "evidence_from_draft": "只保留真实且未被其他章节反驳的证据",\n  "source_sections": ["实际用到的章节"],\n  "retrieval_section_type": "合法section_type",\n  "retrieval_query": "审计后的检索语句",\n  "suggested_direction": "审计后的修改方向",\n  "known_facts": [\n    "从正文确认、且对避免误判很重要的事实"\n  ],\n  "removed_claims": [\n    "原issue中被跨章节证据反驳、以后不得再写入报告的说法"\n  ],\n  "forbidden_phrases": [\n    "最终报告中应避免出现的短语，例如：未说明样本量"\n  ],\n  "audit_note": "为什么keep/corrected/drop"\n}\n"""\n\n\ndef normalize_title(title):\n    return re.sub(\n        r"\\s+",\n        "",\n        title or "",\n    ).lower()\n\n\ndef major_number(title):\n    m = re.match(\n        r"^\\s*(\\d+)(?:\\.\\d+)*",\n        title or "",\n    )\n    if m:\n        return m.group(1)\n    return None\n\n\ndef select_cross_section_scope(\n    issue,\n    sections,\n):\n    selected = []\n    selected_ids = set()\n    matched_indices = []\n\n    def add(sec):\n        sid = sec.get(\n            "section_index"\n        )\n\n        if sid in selected_ids:\n            return\n\n        if sec.get(\n            "section_type"\n        ) in {\n            "front_matter",\n            "reference",\n            "appendix",\n        }:\n            return\n\n        selected.append(sec)\n        selected_ids.add(sid)\n\n    source_titles = issue.get(\n        "source_sections",\n        [],\n    )\n\n    # 1) source section 精确/包含匹配\n    for idx, sec in enumerate(\n        sections\n    ):\n        st = normalize_title(\n            sec["title"]\n        )\n\n        for src in source_titles:\n            ss = normalize_title(src)\n\n            if (\n                ss\n                and (\n                    ss == st\n                    or ss in st\n                    or st in ss\n                )\n            ):\n                add(sec)\n                matched_indices.append(\n                    idx\n                )\n\n    # 2) 同一大章节全部加入，如 5.1 / 5.2 / 5.3\n    majors = {\n        major_number(\n            sections[idx]["title"]\n        )\n        for idx in matched_indices\n    }\n    majors.discard(None)\n\n    for sec in sections:\n        if (\n            major_number(\n                sec["title"]\n            )\n            in majors\n        ):\n            add(sec)\n\n    # 3) source 的相邻章节 ±2，防止跨小节事实遗漏\n    for idx in matched_indices:\n        for j in range(\n            max(0, idx - 2),\n            min(\n                len(sections),\n                idx + 3,\n            ),\n        ):\n            add(\n                sections[j]\n            )\n\n    text = (\n        issue.get(\n            "problem",\n            ""\n        )\n        + " "\n        + issue.get(\n            "reason",\n            ""\n        )\n        + " "\n        + issue.get(\n            "evidence_from_draft",\n            ""\n        )\n    )\n\n    # 4) 关键词触发的跨章节补充\n    if any(\n        x in text.lower()\n        for x in [\n            "saa",\n            "样本平均近似",\n            "升舱",\n        ]\n    ):\n        for sec in sections:\n            if any(\n                x in sec["title"]\n                for x in [\n                    "升舱",\n                    "样本平均近似",\n                ]\n            ):\n                add(sec)\n\n    if any(\n        x in text\n        for x in [\n            "参数",\n            "弹性",\n            "λ",\n            "epsilon",\n            "ε",\n        ]\n    ):\n        for sec in sections:\n            if (\n                sec.get(\n                    "section_type"\n                )\n                == "notation"\n                or "参数" in sec[\n                    "title"\n                ]\n                or "数据证据" in sec[\n                    "title"\n                ]\n                or "敏感性" in sec[\n                    "title"\n                ]\n            ):\n                add(sec)\n\n    if any(\n        x in text\n        for x in [\n            "蒙特卡洛",\n            "验证",\n            "误差",\n            "置信",\n        ]\n    ):\n        for sec in sections:\n            if sec.get(\n                "section_type"\n            ) in {\n                "validation",\n                "sensitivity",\n                "error_analysis",\n            }:\n                add(sec)\n\n    return selected\n\n\ndef build_fact_audit_prompt(\n    issue,\n    sections,\n    outline,\n):\n    scope = (\n        select_cross_section_scope(\n            issue,\n            sections,\n        )\n    )\n\n    blocks = [\n        "===== 论文目录 =====",\n        outline,\n        "",\n        "===== 当前 issue =====",\n        json.dumps(\n            issue,\n            ensure_ascii=False,\n            indent=2,\n        ),\n        "",\n        "===== 跨章节核验正文 =====",\n        (\n            "下列每个区块均为该标题下的完整正文，"\n            "用于核验当前 issue 是否漏看了其他章节已经写明的事实。"\n        ),\n    ]\n\n    total = 0\n\n    for sec in scope:\n        block = (\n            f"\\n\\n【完整章节】{sec[\'title\']}\\n"\n            f"{sec[\'content\'].strip()}\\n"\n            "【章节结束】"\n        )\n\n        if (\n            total + len(block)\n            > FACT_SCOPE_MAX_CHARS\n        ):\n            remain = (\n                FACT_SCOPE_MAX_CHARS\n                - total\n            )\n\n            if remain > 1200:\n                blocks.append(\n                    block[:remain]\n                )\n                blocks.append(\n                    "\\n【注意：最后一个章节因上下文限制未完整提供】"\n                )\n            break\n\n        blocks.append(block)\n        total += len(block)\n\n    return "\\n".join(\n        blocks\n    )\n\n\ndef audit_one_issue(\n    client,\n    issue,\n    sections,\n    outline,\n):\n    prompt = (\n        build_fact_audit_prompt(\n            issue=issue,\n            sections=sections,\n            outline=outline,\n        )\n    )\n\n    data = (\n        v21.reliable_call_json(\n            client=client,\n            model=FACT_AUDIT_MODEL,\n            system_prompt=\n                FACT_AUDIT_SYSTEM,\n            user_prompt=prompt,\n            max_tokens=4500,\n            retries=3,\n        )\n    )\n\n    status = data.get(\n        "status",\n        "keep",\n    )\n\n    if status not in {\n        "keep",\n        "corrected",\n        "drop",\n    }:\n        status = "keep"\n\n    if status == "drop":\n        return None, {\n            "issue_id":\n                issue.get(\n                    "issue_id"\n                ),\n            "problem":\n                issue.get(\n                    "problem"\n                ),\n            "status":\n                "drop",\n            "audit_note":\n                data.get(\n                    "audit_note",\n                    "",\n                ),\n        }\n\n    result = dict(issue)\n\n    # importance/priority 不允许被 audit 升级或重排\n    result[\n        "fact_audit_status"\n    ] = status\n\n    result[\n        "problem"\n    ] = (\n        data.get("problem")\n        or issue.get(\n            "problem",\n            "",\n        )\n    )\n\n    result[\n        "reason"\n    ] = (\n        data.get("reason")\n        or issue.get(\n            "reason",\n            "",\n        )\n    )\n\n    result[\n        "evidence_from_draft"\n    ] = v24.sanitize_evidence(\n        data.get(\n            "evidence_from_draft",\n            issue.get(\n                "evidence_from_draft",\n                "",\n            ),\n        )\n    )\n\n    if data.get(\n        "source_sections"\n    ):\n        result[\n            "source_sections"\n        ] = list(\n            dict.fromkeys(\n                data[\n                    "source_sections"\n                ]\n            )\n        )\n\n    new_type = data.get(\n        "retrieval_section_type"\n    )\n\n    if new_type in (\n        v21.base.SECTION_TYPES\n    ):\n        result[\n            "retrieval_section_type"\n        ] = new_type\n\n    if data.get(\n        "retrieval_query"\n    ):\n        result[\n            "retrieval_query"\n        ] = data[\n            "retrieval_query"\n        ]\n\n    if data.get(\n        "suggested_direction"\n    ):\n        result[\n            "suggested_direction"\n        ] = data[\n            "suggested_direction"\n        ]\n\n    result[\n        "fact_audit"\n    ] = {\n        "status": status,\n        "known_facts":\n            data.get(\n                "known_facts",\n                [],\n            ),\n        "removed_claims":\n            data.get(\n                "removed_claims",\n                [],\n            ),\n        "forbidden_phrases":\n            data.get(\n                "forbidden_phrases",\n                [],\n            ),\n        "audit_note":\n            data.get(\n                "audit_note",\n                "",\n            ),\n    }\n\n    return result, None\n\n\ndef audit_all_issues(\n    client,\n    consolidated_data,\n    sections,\n    outline,\n):\n    print(\n        "\\n[3.8/5] 正在执行 "\n        "Cross-section Fact Audit..."\n    )\n\n    kept = []\n    dropped = []\n\n    issues = consolidated_data.get(\n        "issues",\n        [],\n    )\n\n    for i, issue in enumerate(\n        issues,\n        start=1,\n    ):\n        print(\n            f"\\n[{i}/{len(issues)}] "\n            f"{issue.get(\'issue_id\')} "\n            f"{issue.get(\'problem\')}"\n        )\n\n        result, drop_info = (\n            audit_one_issue(\n                client=client,\n                issue=issue,\n                sections=sections,\n                outline=outline,\n            )\n        )\n\n        if result is not None:\n            kept.append(result)\n            print(\n                "  →",\n                result.get(\n                    "fact_audit_status"\n                ),\n            )\n\n            removed = (\n                result.get(\n                    "fact_audit",\n                    {}\n                ).get(\n                    "removed_claims",\n                    [],\n                )\n            )\n\n            for x in removed:\n                print(\n                    "    删除误判：",\n                    x,\n                )\n        else:\n            dropped.append(\n                drop_info\n            )\n            print(\n                "  → drop"\n            )\n\n    return {\n        "overall_assessment":\n            consolidated_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues": kept,\n        "dropped_by_fact_audit":\n            dropped,\n    }\n\n\n# ============================================================\n# 2. Overlap Audit\n# ============================================================\n\nOVERLAP_SYSTEM = r"""\n你是数学建模评审系统中的“最终重叠裁剪器”。\n\n输入问题已经：\n1. 通过 Evidence Calibration；\n2. 通过初步 Consolidation；\n3. 通过 Cross-section Fact Audit。\n\n因此现在不能找新问题，只能处理“问题之间仍然重复”的情况。\n\n====================\n目标\n====================\n\n让每个子问题只归属于一个最终 issue。\n\n典型情况：\n- 一个 issue 讲 SAA 形式化，\n  另一个 issue 又把“SAA目标函数/决策变量”作为子项；\n  应把该子项只留在一个最合适的 issue 中。\n\n- 一个 issue 讲参数依据，\n  另一个 umbrella issue 又写 λ_i^0 取值依据；\n  应把参数依据留给 parameter issue，\n  umbrella issue 只保留它独有的模型形式化内容。\n\n- 如果一个 umbrella issue 的全部内容都已被其他更具体 issue 覆盖，\n  可以删除该 umbrella issue。\n\n====================\n绝对禁止\n====================\n\n1. 不新增输入中不存在的问题。\n2. 不新增事实。\n3. 不把 fact_audit.removed_claims 恢复回来。\n4. 不出现 fact_audit.forbidden_phrases 对应的错误说法。\n5. importance 不得升级。\n6. high 必须保留。\n7. 不得自行定义符号含义。\n8. 不得编造数值。\n\n====================\n输出\n====================\n\n只输出 JSON：\n\n{\n  "issues": [\n    {\n      "problem": "...",\n      "importance": "high|medium|low",\n      "issue_kind": "...",\n      "reason": "...",\n      "evidence_from_draft": "...",\n      "source_sections": ["..."],\n      "retrieval_section_type": "...",\n      "retrieval_query": "...",\n      "suggested_direction": "...",\n      "merged_from": ["..."],\n      "known_facts": ["..."],\n      "removed_claims": ["..."],\n      "forbidden_phrases": ["..."]\n    }\n  ],\n  "overlap_note": "做了哪些去重"\n}\n"""\n\n\ndef compact_for_overlap(\n    audited_data,\n):\n    result = []\n\n    for issue in audited_data.get(\n        "issues",\n        [],\n    ):\n        fact = issue.get(\n            "fact_audit",\n            {},\n        )\n\n        result.append({\n            "issue_id":\n                issue.get(\n                    "issue_id"\n                ),\n            "problem":\n                issue.get(\n                    "problem",\n                    "",\n                ),\n            "importance":\n                issue.get(\n                    "importance",\n                    "medium",\n                ),\n            "issue_kind":\n                issue.get(\n                    "issue_kind",\n                    (\n                        issue.get(\n                            "evidence_verification",\n                            {}\n                        ).get(\n                            "issue_kind",\n                            "model_completeness",\n                        )\n                    ),\n                ),\n            "reason":\n                issue.get(\n                    "reason",\n                    "",\n                ),\n            "evidence_from_draft":\n                issue.get(\n                    "evidence_from_draft",\n                    "",\n                ),\n            "source_sections":\n                issue.get(\n                    "source_sections",\n                    [],\n                ),\n            "retrieval_section_type":\n                issue.get(\n                    "retrieval_section_type",\n                    "model_detail",\n                ),\n            "retrieval_query":\n                issue.get(\n                    "retrieval_query",\n                    "",\n                ),\n            "suggested_direction":\n                issue.get(\n                    "suggested_direction",\n                    "",\n                ),\n            "merged_from":\n                issue.get(\n                    "merged_from",\n                    [],\n                ),\n            "known_facts":\n                fact.get(\n                    "known_facts",\n                    [],\n                ),\n            "removed_claims":\n                fact.get(\n                    "removed_claims",\n                    [],\n                ),\n            "forbidden_phrases":\n                fact.get(\n                    "forbidden_phrases",\n                    [],\n                ),\n        })\n\n    return result\n\n\ndef normalize_overlap_issue(\n    item,\n):\n    if not isinstance(\n        item,\n        dict,\n    ):\n        return None\n\n    if not item.get(\n        "problem"\n    ):\n        return None\n\n    importance = item.get(\n        "importance",\n        "medium",\n    )\n\n    if importance not in {\n        "high",\n        "medium",\n        "low",\n    }:\n        importance = (\n            "medium"\n        )\n\n    issue_kind = item.get(\n        "issue_kind",\n        "model_completeness",\n    )\n\n    if issue_kind not in {\n        "mathematical_error",\n        "model_completeness",\n        "parameter_basis",\n        "reproducibility",\n        "validation_gap",\n        "presentation",\n    }:\n        issue_kind = (\n            "model_completeness"\n        )\n\n    section_type = item.get(\n        "retrieval_section_type",\n        "model_detail",\n    )\n\n    if section_type not in (\n        v21.base.SECTION_TYPES\n    ):\n        section_type = (\n            "model_detail"\n        )\n\n    return {\n        "problem":\n            item.get(\n                "problem",\n                "",\n            ).strip(),\n        "importance":\n            importance,\n        "issue_kind":\n            issue_kind,\n        "reason":\n            item.get(\n                "reason",\n                "",\n            ).strip(),\n        "evidence_from_draft":\n            v24.sanitize_evidence(\n                item.get(\n                    "evidence_from_draft",\n                    "",\n                )\n            ),\n        "source_sections":\n            list(dict.fromkeys(\n                item.get(\n                    "source_sections",\n                    [],\n                )\n            )),\n        "retrieval_section_type":\n            section_type,\n        "retrieval_query":\n            item.get(\n                "retrieval_query",\n                item.get(\n                    "problem",\n                    "",\n                ),\n            ).strip(),\n        "suggested_direction":\n            item.get(\n                "suggested_direction",\n                "",\n            ).strip(),\n        "merged_from":\n            list(dict.fromkeys(\n                item.get(\n                    "merged_from",\n                    [],\n                )\n            )),\n        "known_facts":\n            list(dict.fromkeys(\n                item.get(\n                    "known_facts",\n                    [],\n                )\n            )),\n        "removed_claims":\n            list(dict.fromkeys(\n                item.get(\n                    "removed_claims",\n                    [],\n                )\n            )),\n        "forbidden_phrases":\n            list(dict.fromkeys(\n                item.get(\n                    "forbidden_phrases",\n                    [],\n                )\n            )),\n    }\n\n\ndef overlap_audit(\n    client,\n    audited_data,\n):\n    print(\n        "\\n[3.9/5] 正在执行 "\n        "Final Overlap Audit..."\n    )\n\n    source = (\n        compact_for_overlap(\n            audited_data\n        )\n    )\n\n    print(\n        "Fact Audit 后问题数：",\n        len(source),\n    )\n\n    prompt = (\n        "请对下面的问题做最终的子问题归属唯一化。\\n"\n        "优先删除纯重复 umbrella issue；"\n        "如果一个 issue 同时含有参数问题和 SAA 问题，"\n        "应把子项分别归入更具体的 issue。\\n"\n        f"最终最多 {MAX_FINAL_ISSUES} 个问题。\\n\\n"\n        + json.dumps(\n            source,\n            ensure_ascii=False,\n            indent=2,\n        )\n    )\n\n    data = (\n        v21.reliable_call_json(\n            client=client,\n            model=OVERLAP_MODEL,\n            system_prompt=\n                OVERLAP_SYSTEM,\n            user_prompt=prompt,\n            max_tokens=8000,\n            retries=3,\n        )\n    )\n\n    issues = []\n\n    for raw in data.get(\n        "issues",\n        [],\n    ):\n        x = (\n            normalize_overlap_issue(\n                raw\n            )\n        )\n\n        if x:\n            issues.append(x)\n\n    issues.sort(\n        key=lambda x:\n            v25.importance_rank(\n                x["importance"]\n            )\n    )\n\n    issues = issues[\n        :MAX_FINAL_ISSUES\n    ]\n\n    # 重新统一 issue_id / priority\n    issues = (\n        v25.assign_issue_ids(\n            issues\n        )\n    )\n\n    # 把 fact lock 信息放回每个 issue\n    for issue in issues:\n        issue[\n            "fact_lock"\n        ] = {\n            "known_facts":\n                issue.pop(\n                    "known_facts",\n                    [],\n                ),\n            "removed_claims":\n                issue.pop(\n                    "removed_claims",\n                    [],\n                ),\n            "forbidden_phrases":\n                issue.pop(\n                    "forbidden_phrases",\n                    [],\n                ),\n        }\n\n    print(\n        "Final Overlap Audit：",\n        len(source),\n        "→",\n        len(issues),\n    )\n\n    for issue in issues:\n        print(\n            f"  {issue[\'issue_id\']} "\n            f"[{issue[\'importance\']}/"\n            f"{issue[\'priority\']}] "\n            f"{issue[\'problem\']}"\n        )\n\n    return {\n        "overall_assessment":\n            audited_data.get(\n                "overall_assessment",\n                "",\n            ),\n        "issues":\n            issues,\n        "overlap_note":\n            data.get(\n                "overlap_note",\n                "",\n            ),\n    }\n\n\n# ============================================================\n# 3. RAG\n# ============================================================\n\ndef retrieve_final_issues(\n    client,\n    final_issue_data,\n    embedding_model,\n    collection,\n):\n    print(\n        "\\n[4/5] 正在为最终独立问题执行 RAG + rerank..."\n    )\n\n    return v23.retrieve_v23(\n        client=client,\n        verified_data=\n            final_issue_data,\n        embedding_model=\n            embedding_model,\n        collection=collection,\n    )\n\n\n# ============================================================\n# 4. Final Report with Claim Lock\n# ============================================================\n\nFINAL_REPORT_SYSTEM = r"""\n你是一名数学建模竞赛论文指导型评审助手。\n\n你收到的 issue 已经经过：\nEvidence Calibration\n→ Consolidation\n→ Cross-section Fact Audit\n→ Final Overlap Audit\n\n所以这是最终允许写入报告的问题集合。\n\n====================\n事实锁\n====================\n\n关于“用户论文当前写了什么/没写什么”的事实，\n只能来自每个 issue 的：\n- evidence_from_draft\n- fact_lock.known_facts\n\n不得自行补充任何论文事实。\n\n尤其禁止：\n\n1. 自行定义符号\n如果 evidence/known_facts 没有明确说明 C0 的含义，\n只能写“C0”，不能自行写成“基准票价”“成本参数”等。\n\n2. 编造数字\n不得出现用户论文没有提供的：\nN=1000、83.2万、4.7%、±20% 等示例数字。\n\n3. 恢复已删除误判\nfact_lock.removed_claims 中的说法已被跨章节证据反驳；\n绝不能重新出现。\nfact_lock.forbidden_phrases 对应的错误说法也不能出现。\n\n4. 把优秀论文案例当成用户论文事实\nRAG 案例中的数字/方法只能写在\n“优秀论文案例怎么处理”中。\n\n====================\n建议允许的范围\n====================\n\n你可以给“如何修改”的建议，包括：\n- 建议补某个公式结构；\n- 建议报告某个统计量；\n- 建议增加某个约束；\n- 建议说明某参数来源。\n\n但必须写成“建议新增/可写为”，\n不能冒充为用户论文已有事实。\n\n建议中的公式优先使用符号形式，\n不要虚构具体数值。\n\n====================\n结构\n====================\n\n只生成前三部分：\n\n# 一、总体评价\n\n# 二、最终独立问题\n\n每个 issue 必须按 issue_id 展开：\n\n## Ixx：问题名（importance / priority）\n\n### 问题是什么\n### 为什么重要\n### 草稿中的核验证据\n### 优秀论文案例怎么处理\n### 从案例抽象什么\n### 作者如何修改\n\n# 三、分模块修改建议\n\n每个模块必须显式标明：\n对应 issue：Ixx\n\n没有对应 issue 的模块不要写。\n\n不要生成第四部分。\n第四部分由程序根据 importance 自动生成。\n"""\n\n\ndef build_locked_reference_text(\n    enriched_data,\n):\n    blocks = []\n\n    for issue in enriched_data.get(\n        "issues",\n        [],\n    ):\n        lock = issue.get(\n            "fact_lock",\n            {},\n        )\n\n        blocks.extend([\n            (\n                f"## {issue[\'issue_id\']} "\n                f"{issue[\'problem\']}"\n            ),\n            (\n                "importance: "\n                f"{issue[\'importance\']}"\n            ),\n            (\n                "priority: "\n                f"{issue[\'priority\']}"\n            ),\n            (\n                "issue_kind: "\n                f"{issue.get(\'issue_kind\')}"\n            ),\n            (\n                "reason: "\n                + issue.get(\n                    "reason",\n                    "",\n                )\n            ),\n            (\n                "evidence_from_draft: "\n                + issue.get(\n                    "evidence_from_draft",\n                    "",\n                )\n            ),\n            (\n                "source_sections: "\n                + ", ".join(\n                    issue.get(\n                        "source_sections",\n                        [],\n                    )\n                )\n            ),\n            (\n                "known_facts: "\n                + json.dumps(\n                    lock.get(\n                        "known_facts",\n                        [],\n                    ),\n                    ensure_ascii=False,\n                )\n            ),\n            (\n                "removed_claims_DO_NOT_USE: "\n                + json.dumps(\n                    lock.get(\n                        "removed_claims",\n                        [],\n                    ),\n                    ensure_ascii=False,\n                )\n            ),\n            (\n                "forbidden_phrases_DO_NOT_USE: "\n                + json.dumps(\n                    lock.get(\n                        "forbidden_phrases",\n                        [],\n                    ),\n                    ensure_ascii=False,\n                )\n            ),\n            (\n                "suggested_direction: "\n                + issue.get(\n                    "suggested_direction",\n                    "",\n                )\n            ),\n        ])\n\n        examples = (\n            issue.get(\n                "retrieval",\n                {}\n            ).get(\n                "examples",\n                [],\n            )\n        )\n\n        if not examples:\n            blocks.append(\n                "RAG examples: none"\n            )\n            continue\n\n        for j, ex in enumerate(\n            examples,\n            start=1,\n        ):\n            blocks.append(\n                f"\\nExample {j}\\n"\n                f"relevance: "\n                f"{ex.get(\'relevance\')}\\n"\n                f"paper_id: "\n                f"{ex.get(\'paper_id\')}\\n"\n                f"section_title: "\n                f"{ex.get(\'section_title\')}\\n"\n                f"rerank_reason: "\n                f"{ex.get(\'reason\')}\\n"\n                f"content:\\n"\n                f"{ex.get(\'content\')}\\n"\n            )\n\n    return "\\n".join(\n        blocks\n    )\n\n\ndef call_report(\n    client,\n    messages,\n    max_tokens,\n):\n    response = (\n        client.chat.completions.create(\n            model=REPORT_MODEL,\n            messages=messages,\n            temperature=0.15,\n            max_tokens=max_tokens,\n            extra_body={\n                "thinking": {\n                    "type":\n                        "disabled"\n                }\n            },\n        )\n    )\n\n    choice = response.choices[0]\n\n    return (\n        (\n            choice.message.content\n            or ""\n        ).strip(),\n        getattr(\n            choice,\n            "finish_reason",\n            None,\n        ),\n    )\n\n\ndef strip_fourth_section(\n    text,\n):\n    marker = re.search(\n        r"(?m)^#\\s*四、",\n        text or "",\n    )\n\n    if marker:\n        return (\n            text[\n                :marker.start()\n            ].rstrip()\n        )\n\n    return (\n        text or ""\n    ).rstrip()\n\n\ndef generate_report_draft(\n    client,\n    outline,\n    enriched_data,\n):\n    reference_text = (\n        build_locked_reference_text(\n            enriched_data\n        )\n    )\n\n    user_prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline}\\n\\n"\n        "===== 最终事实锁定 issue =====\\n"\n        f"{reference_text}\\n\\n"\n        "只能依据上述 evidence/known_facts "\n        "描述用户论文现状。"\n    )\n\n    messages = [\n        {\n            "role": "system",\n            "content":\n                FINAL_REPORT_SYSTEM,\n        },\n        {\n            "role": "user",\n            "content":\n                user_prompt,\n        },\n    ]\n\n    content, finish_reason = (\n        call_report(\n            client,\n            messages,\n            REPORT_MAX_TOKENS,\n        )\n    )\n\n    if not content:\n        raise RuntimeError(\n            "V2.5.1 报告为空"\n        )\n\n    if finish_reason == "length":\n        print(\n            "警告：报告达到长度限制，"\n            "建议将 MAX_TOKENS 调高。"\n        )\n\n    return strip_fourth_section(\n        content\n    )\n\n\n# ============================================================\n# 5. Report factual-claim audit\n# ============================================================\n\nREPORT_AUDIT_SYSTEM = r"""\n你是最终报告的“事实越权审计器”。\n\n输入：\n1. 最终允许的 issue（含 evidence_from_draft、known_facts、\n   removed_claims、forbidden_phrases）\n2. 已生成报告\n\n只检查“报告对用户论文现状的事实陈述”。\n\n判为 violation 的情况：\n- 报告说某内容“没写”，但 known_facts 显示其他章节已经写了；\n- 报告恢复了 removed_claims；\n- 报告出现 forbidden_phrases 对应的错误断言；\n- 报告自行定义了 evidence 没定义的符号；\n- 报告虚构了用户论文不存在的数字；\n- 把知识库优秀论文中的数字说成用户论文自己的数字。\n\n不判 violation：\n- 明确标记为“建议”的符号化公式；\n- 明确标记为优秀论文案例的数字；\n- 对 evidence 的合理同义改写。\n\n输出：\n{\n  "pass": true,\n  "violations": [\n    {\n      "text": "报告中的问题句",\n      "reason": "为什么越权",\n      "repair": "应该怎么改"\n    }\n  ]\n}\n"""\n\n\ndef compact_fact_lock(\n    enriched_data,\n):\n    items = []\n\n    for issue in enriched_data.get(\n        "issues",\n        [],\n    ):\n        items.append({\n            "issue_id":\n                issue.get(\n                    "issue_id"\n                ),\n            "problem":\n                issue.get(\n                    "problem"\n                ),\n            "evidence_from_draft":\n                issue.get(\n                    "evidence_from_draft",\n                    "",\n                ),\n            "fact_lock":\n                issue.get(\n                    "fact_lock",\n                    {},\n                ),\n        })\n\n    return items\n\n\ndef audit_report_claims(\n    client,\n    report_first_three,\n    enriched_data,\n):\n    prompt = (\n        "===== 允许事实 =====\\n"\n        + json.dumps(\n            compact_fact_lock(\n                enriched_data\n            ),\n            ensure_ascii=False,\n            indent=2,\n        )\n        + "\\n\\n===== 待审报告 =====\\n"\n        + report_first_three\n    )\n\n    return v21.reliable_call_json(\n        client=client,\n        model=REPORT_AUDIT_MODEL,\n        system_prompt=\n            REPORT_AUDIT_SYSTEM,\n        user_prompt=prompt,\n        max_tokens=5000,\n        retries=3,\n    )\n\n\ndef repair_report(\n    client,\n    bad_report,\n    violations,\n    outline,\n    enriched_data,\n):\n    reference_text = (\n        build_locked_reference_text(\n            enriched_data\n        )\n    )\n\n    repair_system = (\n        FINAL_REPORT_SYSTEM\n        + "\\n\\n"\n        + r"""\n你现在是在修复一份已经生成的报告。\n必须逐条修复 fact audit 给出的 violation。\n除修复越权事实外，不要增加任何新问题。\n"""\n    )\n\n    prompt = (\n        "===== 论文目录 =====\\n"\n        f"{outline}\\n\\n"\n        "===== 最终事实锁定 issue =====\\n"\n        f"{reference_text}\\n\\n"\n        "===== 原报告 =====\\n"\n        f"{bad_report}\\n\\n"\n        "===== Fact Audit violations =====\\n"\n        + json.dumps(\n            violations,\n            ensure_ascii=False,\n            indent=2,\n        )\n        + "\\n\\n请输出修复后的前三部分完整报告。"\n    )\n\n    content, _ = call_report(\n        client,\n        [\n            {\n                "role": "system",\n                "content":\n                    repair_system,\n            },\n            {\n                "role": "user",\n                "content":\n                    prompt,\n            },\n        ],\n        REPORT_REPAIR_MAX_TOKENS,\n    )\n\n    return strip_fourth_section(\n        content\n    )\n\n\ndef generate_final_report(\n    client,\n    outline,\n    enriched_data,\n):\n    print(\n        "\\n[5/5] 正在生成 "\n        "V2.5.1 事实锁定报告..."\n    )\n\n    first_three = (\n        generate_report_draft(\n            client=client,\n            outline=outline,\n            enriched_data=\n                enriched_data,\n        )\n    )\n\n    print(\n        "正在做最终 factual-claim audit..."\n    )\n\n    audit = (\n        audit_report_claims(\n            client=client,\n            report_first_three=\n                first_three,\n            enriched_data=\n                enriched_data,\n        )\n    )\n\n    if not audit.get(\n        "pass",\n        False,\n    ):\n        violations = (\n            audit.get(\n                "violations",\n                [],\n            )\n        )\n\n        print(\n            f"发现 {len(violations)} 个"\n            "事实越权点，自动修复一次..."\n        )\n\n        for item in violations:\n            print(\n                "  -",\n                item.get(\n                    "reason",\n                    "",\n                ),\n            )\n\n        first_three = (\n            repair_report(\n                client=client,\n                bad_report=\n                    first_three,\n                violations=\n                    violations,\n                outline=outline,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        # 修复后再审一次，仅记录，不无限循环\n        audit2 = (\n            audit_report_claims(\n                client=client,\n                report_first_three=\n                    first_three,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        if not audit2.get(\n            "pass",\n            False,\n        ):\n            print(\n                "警告：修复后仍存在 factual audit "\n                "疑点，请把 06_report_fact_audit_v251.json "\n                "发给我。"\n            )\n            audit = audit2\n        else:\n            audit = audit2\n\n    priority = (\n        v25.generate_priority_section(\n            enriched_data.get(\n                "issues",\n                [],\n            )\n        )\n    )\n\n    final = (\n        first_three.rstrip()\n        + "\\n\\n"\n        + priority\n        + "\\n"\n    )\n\n    return final.strip(), audit\n\n\n# ============================================================\n# Main\n# ============================================================\n\ndef main():\n    api_key = (\n        v21.base.require_api_key()\n    )\n\n    if len(sys.argv) < 2:\n        print(\n            "用法：\\n"\n            "python "\n            "review_agent_deepseek_v2_5_1.py "\n            "\\"你的论文.md\\""\n        )\n        sys.exit(1)\n\n    paper_path, text = (\n        v21.base.read_paper(\n            sys.argv[1]\n        )\n    )\n\n    print("=" * 72)\n    print(\n        "数学建模整篇论文评审 Agent V2.5.1"\n    )\n    print("=" * 72)\n    print(\n        "论文：",\n        paper_path,\n    )\n    print(\n        "Fact Audit模型：",\n        FACT_AUDIT_MODEL,\n    )\n    print(\n        "Overlap模型：",\n        OVERLAP_MODEL,\n    )\n    print(\n        "Report模型：",\n        REPORT_MODEL,\n    )\n\n    work_dir = (\n        v21.make_work_dir(\n            paper_path,\n            text,\n        )\n    )\n\n    sections = (\n        v21.split_sections_v21(\n            text\n        )\n    )\n\n    outline = (\n        v24.build_clean_outline(\n            sections\n        )\n    )\n\n    client = (\n        v21.base.make_client(\n            api_key\n        )\n    )\n\n    # --------------------------------------------------------\n    # 直接复用 V2.5 consolidation\n    # --------------------------------------------------------\n\n    v25_consolidated_path = (\n        work_dir\n        / "04c_consolidated_issues_v25.json"\n    )\n\n    consolidated_v25 = load_json(\n        v25_consolidated_path\n    )\n\n    if consolidated_v25 is None:\n        print(\n            "没有发现 "\n            "04c_consolidated_issues_v25.json。"\n        )\n        print(\n            "请先运行 V2.5。"\n        )\n        sys.exit(1)\n\n    print(\n        "[1~3.7/5] 检测到 V2.5 consolidation "\n        "缓存，直接复用。"\n    )\n\n    # --------------------------------------------------------\n    # Cross-section Fact Audit\n    # --------------------------------------------------------\n\n    fact_audit_path = (\n        work_dir\n        / "04d_cross_section_fact_audit_v251.json"\n    )\n\n    audited_data = load_json(\n        fact_audit_path\n    )\n\n    if audited_data is None:\n        audited_data = (\n            audit_all_issues(\n                client=client,\n                consolidated_data=\n                    consolidated_v25,\n                sections=sections,\n                outline=outline,\n            )\n        )\n\n        save_json(\n            fact_audit_path,\n            audited_data,\n        )\n    else:\n        print(\n            "[3.8/5] 检测到 Cross-section "\n            "Fact Audit 缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Final Overlap Audit\n    # --------------------------------------------------------\n\n    final_issue_path = (\n        work_dir\n        / "04e_final_issues_v251.json"\n    )\n\n    final_issue_data = (\n        load_json(\n            final_issue_path\n        )\n    )\n\n    if final_issue_data is None:\n        final_issue_data = (\n            overlap_audit(\n                client=client,\n                audited_data=\n                    audited_data,\n            )\n        )\n\n        save_json(\n            final_issue_path,\n            final_issue_data,\n        )\n    else:\n        print(\n            "[3.9/5] 检测到 Final Overlap "\n            "Audit 缓存，直接复用。"\n        )\n\n    if not final_issue_data.get(\n        "issues"\n    ):\n        print(\n            "最终没有问题保留，停止。"\n        )\n        return\n\n    # --------------------------------------------------------\n    # RAG\n    # --------------------------------------------------------\n\n    retrieval_path = (\n        work_dir\n        / "05_retrieval_v251.json"\n    )\n\n    enriched_data = load_json(\n        retrieval_path\n    )\n\n    if enriched_data is None:\n        collection, embedding_model = (\n            v21.base.initialize_retrieval()\n        )\n\n        enriched_data = (\n            retrieve_final_issues(\n                client=client,\n                final_issue_data=\n                    final_issue_data,\n                embedding_model=\n                    embedding_model,\n                collection=collection,\n            )\n        )\n\n        save_json(\n            retrieval_path,\n            enriched_data,\n        )\n    else:\n        print(\n            "[4/5] 检测到 V2.5.1 RAG 缓存，直接复用。"\n        )\n\n    # --------------------------------------------------------\n    # Report\n    # --------------------------------------------------------\n\n    report_path = (\n        work_dir\n        / "review_report_v251.md"\n    )\n\n    report_audit_path = (\n        work_dir\n        / "06_report_fact_audit_v251.json"\n    )\n\n    if report_path.exists():\n        report = (\n            report_path.read_text(\n                encoding="utf-8",\n                errors="ignore",\n            )\n        )\n\n        if v25.report_complete(\n            report,\n            enriched_data.get(\n                "issues",\n                [],\n            ),\n        ):\n            print(\n                "[5/5] 检测到完整 V2.5.1 报告，直接复用。"\n            )\n        else:\n            report, report_audit = (\n                generate_final_report(\n                    client=client,\n                    outline=outline,\n                    enriched_data=\n                        enriched_data,\n                )\n            )\n\n            report_path.write_text(\n                report,\n                encoding="utf-8",\n            )\n\n            save_json(\n                report_audit_path,\n                report_audit,\n            )\n    else:\n        report, report_audit = (\n            generate_final_report(\n                client=client,\n                outline=outline,\n                enriched_data=\n                    enriched_data,\n            )\n        )\n\n        report_path.write_text(\n            report,\n            encoding="utf-8",\n        )\n\n        save_json(\n            report_audit_path,\n            report_audit,\n        )\n\n    print()\n    print("=" * 72)\n    print(\n        "V2.5.1 最终报告"\n    )\n    print("=" * 72)\n    print()\n    print(report)\n\n    print()\n    print("=" * 72)\n    print(\n        "Cross-section Fact Audit：",\n        fact_audit_path,\n    )\n    print(\n        "Final Issues：",\n        final_issue_path,\n    )\n    print(\n        "RAG：",\n        retrieval_path,\n    )\n    print(\n        "Report Fact Audit：",\n        report_audit_path,\n    )\n    print(\n        "最终报告：",\n        report_path,\n    )\n\n\nif __name__ == "__main__":\n    main()\n'
review_agent_deepseek_v2_5_1 = _load_embedded_module('review_agent_deepseek_v2_5_1', _SOURCE_REVIEW_AGENT_DEEPSEEK_V2_5_1)


# ============================================================
# Public entry point: Agent V3
# Rubric V1 + Experience Library V1 + frozen V2.5.1 pipeline
# ============================================================

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# Short aliases to the already-loaded frozen modules.
v21 = review_agent_deepseek_v2_1
v23 = review_agent_deepseek_v2_3
v24 = review_agent_deepseek_v2_4
v25 = review_agent_deepseek_v2_5
v251 = review_agent_deepseek_v2_5_1


V3_ROOT = Path("D:/math_model_kb")

RUBRIC_V1_PATH = (
    V3_ROOT / "rubric_v1.json"
)

EXPERIENCE_V1_PATH = (
    V3_ROOT
    / "experience_library_v1_frozen.jsonl"
)

RUBRIC_EXPERIENCE_MODEL = os.getenv(
    "DEEPSEEK_RUBRIC_EXPERIENCE_MODEL",
    "deepseek-v4-flash",
)

EXPERIENCES_PER_ISSUE = 3

# V3 layer cache/version marker.
# Changing this value invalidates 04f/05/report V3 caches while preserving
# the expensive V2.5.1 diagnosis/evidence/fact-audit checkpoints.
V3_LAYER_VERSION = "3.0.1"


# ============================================================
# Basic helpers
# ============================================================

def _load_json_file(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def _save_json_file(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{path} 第 {line_no} 行 "
                    f"JSON 解析失败：{exc}"
                )

    return rows


def _file_sha12(path):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()[:12]


def _issue_sha12(data):
    raw = json.dumps(
        data.get(
            "issues",
            [],
        ),
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:12]


def _load_cache_with_signature(
    path,
    signature,
):
    if not path.exists():
        return None

    try:
        data = _load_json_file(
            path
        )
    except Exception:
        return None

    if (
        data.get(
            "v3_signature"
        )
        != signature
    ):
        return None

    return data


# ============================================================
# Load frozen Rubric / Experience Library
# ============================================================

def load_v3_knowledge():
    if not RUBRIC_V1_PATH.exists():
        raise FileNotFoundError(
            "找不到 Rubric V1：\n"
            f"{RUBRIC_V1_PATH}"
        )

    if not EXPERIENCE_V1_PATH.exists():
        raise FileNotFoundError(
            "找不到 Experience Library V1 冻结版：\n"
            f"{EXPERIENCE_V1_PATH}\n\n"
            "请确认已经把 "
            "experience_library_v1_frozen.jsonl "
            "放到 D:/math_model_kb/"
        )

    rubric = _load_json_file(
        RUBRIC_V1_PATH
    )

    experiences = _read_jsonl(
        EXPERIENCE_V1_PATH
    )

    dims = rubric.get(
        "dimensions",
        [],
    )

    if len(dims) != 12:
        raise RuntimeError(
            "rubric_v1.json 不是预期的 12 维 Rubric。"
        )

    rubric_ids = {
        x.get(
            "rubric_id"
        )
        for x in dims
    }

    expected_ids = {
        f"R{i:02d}"
        for i in range(
            1,
            13,
        )
    }

    if rubric_ids != expected_ids:
        raise RuntimeError(
            "Rubric ID 不完整，应为 R01~R12。"
        )

    exp_ids = [
        x.get(
            "experience_id"
        )
        for x in experiences
    ]

    if (
        not experiences
        or len(exp_ids)
        != len(
            set(exp_ids)
        )
        or any(
            not x
            for x in exp_ids
        )
    ):
        raise RuntimeError(
            "Experience Library experience_id "
            "为空或重复。"
        )

    for row in experiences:
        if (
            row.get(
                "rubric_id"
            )
            not in expected_ids
        ):
            raise RuntimeError(
                "Experience Library 中存在非法 rubric_id："
                f"{row.get('rubric_id')}"
            )

        sources = row.get(
            "representative_sources",
            [],
        )

        if not sources:
            raise RuntimeError(
                "Experience Library 中存在空证据经验："
                f"{row.get('experience_id')}"
            )

    knowledge_signature = (
        _file_sha12(
            RUBRIC_V1_PATH
        )
        + "-"
        + _file_sha12(
            EXPERIENCE_V1_PATH
        )
    )

    return (
        rubric,
        experiences,
        knowledge_signature,
    )


def compact_rubric_catalog(
    rubric,
):
    return [
        {
            "rubric_id":
                x["rubric_id"],
            "name":
                x["name"],
            "description":
                x.get(
                    "description",
                    "",
                ),
            "checks":
                x.get(
                    "checks",
                    [],
                )[:6],
        }
        for x in rubric[
            "dimensions"
        ]
    ]


def compact_experience_catalog(
    experiences,
):
    result = []

    for x in experiences:
        result.append({
            "experience_id":
                x[
                    "experience_id"
                ],
            "rubric_id":
                x[
                    "rubric_id"
                ],
            "pattern":
                x.get(
                    "pattern",
                    "",
                ),
            "problem_solved":
                x.get(
                    "problem_solved",
                    "",
                ),
            "applicable_when":
                x.get(
                    "applicable_when",
                    "",
                ),
            "evidence_strength":
                x.get(
                    "evidence_strength",
                    "",
                ),
            "support_count":
                x.get(
                    "support_count",
                    0,
                ),
            "tags":
                x.get(
                    "tags",
                    [],
                ),
        })

    return result


# ============================================================
# Rubric + Experience mapping
# ============================================================

RUBRIC_EXPERIENCE_SYSTEM = r"""
你是“数学建模论文评审 Agent 的 Rubric + Experience 路由器”。

输入中的问题已经经过：
Evidence Calibration
→ Post-Evidence Consolidation
→ Cross-section Fact Audit
→ Final Overlap Audit

因此：
【不得新增论文问题】
【不得恢复已经删除的问题】
【不得改变草稿事实】

你的任务只有两个：

1. 把每个最终 issue 映射到内部评审 Rubric。
2. 从给定 Experience Library 中选择真正能指导该 issue 修改的可迁移经验。

Rubric 是内部、非官方评审框架：
R01 摘要
R02 问题重述与问题分析
R03 模型假设
R04 符号与变量定义
R05 数据处理与参数依据
R06 模型选择依据
R07 模型建立完整性
R08 模型求解与可复现性
R09 结果分析
R10 模型验证、误差、灵敏度与鲁棒性
R11 模型评价与推广
R12 结论与全文呈现

规则：

1. primary_rubric_id 必须且只能有一个。
2. secondary_rubric_ids 最多 1 个；没有则 []。
3. 不允许给任何数值评分，也不允许称其为官方评分标准。
4. experience_ids 最多 3 条。
5. 经验必须来自输入给定的 experience_id。
6. 优先选：
   - 与当前问题核心缺陷直接相关；
   - 可明确指导修改结构；
   - repeated / support_count 较高的经验。
7. 不要因为同属一个 Rubric 就强行选经验。
   只有当经验直接针对当前 issue 的核心缺陷时才选择。
   如果只是“同属一个Rubric”“原则上有一点类似”
   “可以类比参考但并不解决同一问题”，都视为不够直接，
   experience_ids 必须返回 []。
   没有直接合适经验是正常结果，不需要为了完整性凑经验。
8. 经验库是“优秀论文中抽象出的可迁移经验”，
   不能把经验案例里的具体数字当作用户论文应该照抄的规则。
9. rag_query 用于下一阶段搜索优秀论文真实案例：
   - 应围绕当前 issue 的问题本质；
   - 可以吸收选中经验的通用做法；
   - 不要虚构用户论文不存在的事实；
   - 不要写具体优秀论文 paper_id；
   - 保持为一句适合中文语义检索的查询。

只输出合法 JSON object：

{
  "issues": [
    {
      "issue_id": "I01",
      "primary_rubric_id": "R07",
      "rubric_reason": "为什么属于该Rubric",
      "secondary_rubric_ids": [],
      "experience_ids": [
        "EXP_R07_001"
      ],
      "experience_reason": "这些经验为什么适合当前问题",
      "rag_query": "用于优秀论文知识库检索真实案例的中文查询"
    }
  ]
}

必须覆盖输入中的每个 issue_id。
"""


def default_rubric_id(
    issue,
):
    text = " ".join([
        str(
            issue.get(
                "problem",
                "",
            )
        ),
        str(
            issue.get(
                "reason",
                "",
            )
        ),
        str(
            issue.get(
                "retrieval_section_type",
                "",
            )
        ),
    ])

    keyword_rules = [
        (
            "R01",
            [
                "摘要",
            ],
        ),
        (
            "R12",
            [
                "结论",
                "总结",
                "全文呈现",
                "跨问题",
                "全文衔接",
            ],
        ),
        (
            "R10",
            [
                "验证",
                "检验",
                "误差",
                "灵敏",
                "敏感性",
                "鲁棒",
                "蒙特卡洛",
                "置信区间",
            ],
        ),
        (
            "R11",
            [
                "模型评价",
                "优缺点",
                "推广",
                "局限",
            ],
        ),
        (
            "R08",
            [
                "求解",
                "复现",
                "算法参数",
                "迭代",
                "收敛",
                "求解器",
            ],
        ),
        (
            "R06",
            [
                "模型选择",
                "模型比较",
                "选择依据",
                "候选模型",
            ],
        ),
        (
            "R05",
            [
                "参数依据",
                "参数来源",
                "数据处理",
                "预处理",
                "缺失",
                "异常值",
                "数据来源",
            ],
        ),
        (
            "R04",
            [
                "符号",
                "变量定义",
                "下标",
            ],
        ),
        (
            "R03",
            [
                "模型假设",
                "假设",
            ],
        ),
        (
            "R07",
            [
                "约束",
                "目标函数",
                "决策变量",
                "模型建立",
                "形式化",
                "完整性",
            ],
        ),
        (
            "R09",
            [
                "结果分析",
                "结果解释",
                "基线",
                "结果比较",
            ],
        ),
        (
            "R02",
            [
                "问题分析",
                "问题重述",
                "任务拆解",
                "技术路线",
            ],
        ),
    ]

    for rid, words in keyword_rules:
        if any(
            w in text
            for w in words
        ):
            return rid

    section_map = {
        "abstract": "R01",
        "background": "R02",
        "problem_statement": "R02",
        "analysis": "R02",
        "assumption": "R03",
        "notation": "R04",
        "preprocessing": "R05",
        "model_selection": "R06",
        "model_idea": "R06",
        "model_building": "R07",
        "model_detail": "R07",
        "model_solving": "R08",
        "result": "R09",
        "validation": "R10",
        "error_analysis": "R10",
        "sensitivity": "R10",
        "evaluation": "R11",
        "strength": "R11",
        "weakness": "R11",
        "generalization": "R11",
    }

    return section_map.get(
        issue.get(
            "retrieval_section_type"
        ),
        "R07",
    )


def _char_bigrams(text):
    s = re.sub(
        r"\s+",
        "",
        str(text or ""),
    )

    s = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9]",
        "",
        s,
    )

    if len(s) < 2:
        return {
            s
        } if s else set()

    return {
        s[i:i + 2]
        for i in range(
            len(s) - 1
        )
    }


def fallback_experience_ids(
    issue,
    rubric_id,
    experiences,
):
    query_text = " ".join([
        issue.get(
            "problem",
            "",
        ),
        issue.get(
            "reason",
            "",
        ),
        issue.get(
            "suggested_direction",
            "",
        ),
    ])

    q = _char_bigrams(
        query_text
    )

    scored = []

    for exp in experiences:
        if (
            exp.get(
                "rubric_id"
            )
            != rubric_id
        ):
            continue

        e_text = " ".join([
            exp.get(
                "pattern",
                "",
            ),
            exp.get(
                "problem_solved",
                "",
            ),
            " ".join(
                exp.get(
                    "tags",
                    [],
                )
            ),
        ])

        e = _char_bigrams(
            e_text
        )

        overlap = len(
            q & e
        )

        repeated_bonus = (
            2
            if exp.get(
                "evidence_strength"
            )
            == "repeated"
            else 0
        )

        support_bonus = min(
            int(
                exp.get(
                    "support_count",
                    0,
                )
            ),
            5,
        ) * 0.15

        score = (
            overlap
            + repeated_bonus
            + support_bonus
        )

        scored.append(
            (
                score,
                exp[
                    "experience_id"
                ],
            )
        )

    scored.sort(
        reverse=True
    )

    return [
        eid
        for score, eid in scored[
            :EXPERIENCES_PER_ISSUE
        ]
        if score > 0
    ]


def compact_selected_experience(
    exp,
):
    sources = []

    for source in exp.get(
        "representative_sources",
        [],
    )[:2]:
        excerpt = str(
            source.get(
                "excerpt",
                "",
            )
            or ""
        ).strip()

        if len(excerpt) > 800:
            excerpt = (
                excerpt[:800]
                + "..."
            )

        sources.append({
            "paper_id":
                source.get(
                    "paper_id"
                ),
            "title":
                source.get(
                    "title"
                ),
            "section_type":
                source.get(
                    "section_type"
                ),
            "excerpt":
                excerpt,
        })

    return {
        "experience_id":
            exp[
                "experience_id"
            ],
        "rubric_id":
            exp[
                "rubric_id"
            ],
        "pattern":
            exp.get(
                "pattern",
                "",
            ),
        "problem_solved":
            exp.get(
                "problem_solved",
                "",
            ),
        "why_effective":
            exp.get(
                "why_effective",
                "",
            ),
        "applicable_when":
            exp.get(
                "applicable_when",
                "",
            ),
        "recommended_structure":
            exp.get(
                "recommended_structure",
                [],
            )[:6],
        "do_not_overgeneralize":
            exp.get(
                "do_not_overgeneralize",
                "",
            ),
        "support_count":
            exp.get(
                "support_count",
                0,
            ),
        "evidence_strength":
            exp.get(
                "evidence_strength",
                "",
            ),
        "source_papers":
            exp.get(
                "source_papers",
                [],
            ),
        "representative_sources":
            sources,
    }


def attach_rubric_and_experience(
    client,
    final_issue_data,
    rubric,
    experiences,
):
    print(
        "\n[4.0/6] 正在执行 "
        "Rubric V1 + Experience Library V1 路由..."
    )

    issues = final_issue_data.get(
        "issues",
        [],
    )

    if not issues:
        return {
            "overall_assessment":
                final_issue_data.get(
                    "overall_assessment",
                    "",
                ),
            "issues": [],
        }

    compact_issues = []

    for issue in issues:
        compact_issues.append({
            "issue_id":
                issue.get(
                    "issue_id"
                ),
            "problem":
                issue.get(
                    "problem",
                    "",
                ),
            "importance":
                issue.get(
                    "importance",
                    "",
                ),
            "reason":
                issue.get(
                    "reason",
                    "",
                ),
            "evidence_from_draft":
                issue.get(
                    "evidence_from_draft",
                    "",
                ),
            "retrieval_section_type":
                issue.get(
                    "retrieval_section_type",
                    "",
                ),
            "retrieval_query":
                issue.get(
                    "retrieval_query",
                    "",
                ),
            "suggested_direction":
                issue.get(
                    "suggested_direction",
                    "",
                ),
            "known_facts":
                issue.get(
                    "fact_lock",
                    {},
                ).get(
                    "known_facts",
                    [],
                ),
        })

    payload = {
        "rubric_catalog":
            compact_rubric_catalog(
                rubric
            ),
        "experience_catalog":
            compact_experience_catalog(
                experiences
            ),
        "final_issues":
            compact_issues,
    }

    data = v21.reliable_call_json(
        client=client,
        model=RUBRIC_EXPERIENCE_MODEL,
        system_prompt=
            RUBRIC_EXPERIENCE_SYSTEM,
        user_prompt=(
            "请完成以下最终 issue 的 "
            "Rubric 映射与经验选择。\n"
            "只输出合法 JSON object。\n\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
            )
        ),
        max_tokens=9500,
        retries=3,
    )

    choice_map = {}

    for row in data.get(
        "issues",
        [],
    ):
        if not isinstance(
            row,
            dict,
        ):
            continue

        iid = str(
            row.get(
                "issue_id",
                "",
            )
        ).strip()

        if iid:
            choice_map[
                iid
            ] = row

    rubric_map = {
        x["rubric_id"]: x
        for x in rubric[
            "dimensions"
        ]
    }

    exp_map = {
        x["experience_id"]: x
        for x in experiences
    }

    enriched = []

    for issue in issues:
        iid = issue.get(
            "issue_id"
        )

        choice = choice_map.get(
            iid,
            {},
        )

        primary = str(
            choice.get(
                "primary_rubric_id",
                "",
            )
        ).strip()

        if primary not in rubric_map:
            primary = (
                default_rubric_id(
                    issue
                )
            )

        secondary = (
            choice.get(
                "secondary_rubric_ids",
                [],
            )
        )

        if not isinstance(
            secondary,
            list,
        ):
            secondary = []

        secondary = [
            str(x).strip()
            for x in secondary
            if (
                str(x).strip()
                in rubric_map
                and str(x).strip()
                != primary
            )
        ][:1]

        allowed_rubrics = {
            primary,
            *secondary,
        }

        selected_ids = (
            choice.get(
                "experience_ids",
                [],
            )
        )

        if not isinstance(
            selected_ids,
            list,
        ):
            selected_ids = []

        valid_ids = []

        for eid in selected_ids:
            eid = str(
                eid
            ).strip()

            exp = exp_map.get(
                eid
            )

            if not exp:
                continue

            if (
                exp.get(
                    "rubric_id"
                )
                not in allowed_rubrics
            ):
                continue

            if eid in valid_ids:
                continue

            valid_ids.append(
                eid
            )

            if (
                len(valid_ids)
                >= EXPERIENCES_PER_ISSUE
            ):
                break

        # Do NOT force a same-Rubric fallback.
        # An empty experience match is preferable to a weak analogy.
        # The raw excellent-paper RAG still runs for every issue.
        if not valid_ids:
            valid_ids = []

        guidance = [
            compact_selected_experience(
                exp_map[eid]
            )
            for eid in valid_ids
            if eid in exp_map
        ]

        original_query = (
            issue.get(
                "retrieval_query",
                "",
            )
        ).strip()

        rag_query = str(
            choice.get(
                "rag_query",
                "",
            )
            or ""
        ).strip()

        if not rag_query:
            patterns = [
                x.get(
                    "pattern",
                    "",
                )
                for x in guidance[
                    :2
                ]
                if x.get(
                    "pattern"
                )
            ]

            rag_query = (
                original_query
            )

            if patterns:
                rag_query += (
                    "；优秀数学建模论文中如何"
                    + "；".join(
                        patterns
                    )
                )

        # 控制检索query长度。
        if len(rag_query) > 650:
            rag_query = (
                rag_query[:650]
            )

        x = dict(issue)

        dim = rubric_map[
            primary
        ]

        x["rubric"] = {
            "rubric_id":
                primary,
            "rubric_name":
                dim["name"],
            "description":
                dim.get(
                    "description",
                    "",
                ),
            "reason":
                str(
                    choice.get(
                        "rubric_reason",
                        "",
                    )
                    or ""
                ).strip(),
            "secondary_rubric_ids":
                secondary,
        }

        x[
            "experience_guidance"
        ] = guidance

        x[
            "experience_selection_reason"
        ] = str(
            choice.get(
                "experience_reason",
                "",
            )
            or ""
        ).strip()

        x[
            "retrieval_query_original"
        ] = original_query

        x[
            "retrieval_query_v3"
        ] = rag_query

        enriched.append(x)

        print(
            f"  {iid} → "
            f"{primary} {dim['name']}；"
            f"经验 {len(guidance)} 条"
        )

    return {
        "overall_assessment":
            final_issue_data.get(
                "overall_assessment",
                "",
            ),
        "issues":
            enriched,
        "rubric_framework": {
            "name":
                rubric.get(
                    "metadata",
                    {}
                ).get(
                    "name",
                    "Math Modeling Paper Review Rubric",
                ),
            "version":
                rubric.get(
                    "metadata",
                    {}
                ).get(
                    "version",
                    "1.0",
                ),
            "official":
                False,
            "note":
                "内部非官方评审框架，不进行官方分数声称。",
        },
    }


# ============================================================
# RAG with Experience-enhanced query
# ============================================================

def retrieve_v3(
    client,
    rubric_experience_data,
    embedding_model,
    collection,
):
    rag_input = {
        "overall_assessment":
            rubric_experience_data.get(
                "overall_assessment",
                "",
            ),
        "issues": [],
    }

    for issue in rubric_experience_data.get(
        "issues",
        [],
    ):
        x = dict(issue)

        # Only the query changes.
        # Existing V2.5.1 RAG/rerank remains untouched.
        x["retrieval_query"] = (
            x.get(
                "retrieval_query_v3"
            )
            or x.get(
                "retrieval_query"
            )
        )

        rag_input[
            "issues"
        ].append(x)

    result = (
        v251.retrieve_final_issues(
            client=client,
            final_issue_data=
                rag_input,
            embedding_model=
                embedding_model,
            collection=collection,
        )
    )

    return result


# ============================================================
# V3 report prompt
# ============================================================

V3_FINAL_REPORT_SYSTEM = r"""
你是一名数学建模竞赛论文“指导型”评审助手。

输入中的 issue 已经经过：

局部/全文诊断
→ Evidence Calibration
→ Post-Evidence Consolidation
→ Cross-section Fact Audit
→ Final Overlap Audit
→ Rubric V1 映射
→ Experience Library V1 匹配
→ 优秀论文原始 RAG 检索

Rubric V1 是内部、非官方评审框架。
绝对不要声称它是全国大学生数学建模竞赛官方评分标准，
也不要给出虚构的官方分数或权重。

==================================================
事实锁
==================================================

关于“用户论文已经写了什么 / 没写什么”的事实，
只能来自：

- evidence_from_draft
- fact_lock.known_facts

不得自行补充用户论文事实。

特别禁止：

1. evidence 没定义的符号，不得自行定义。
2. 不得编造用户论文不存在的数字。
3. removed_claims / forbidden_phrases 中的误判不得恢复。
4. Experience Library 和 RAG 中的内容属于优秀论文外部案例，
   绝不能写成用户论文自己的内容。
5. 在“作者如何修改”和“分模块修改建议”中，也不得替作者虚构：
   - 实际使用的软件、编程语言、库、求解器或算法；
   - 随机种子的具体数值；
   - 运行时间、最优性间隙、收敛轮数；
   - 用户论文未给出的参数值、阈值、比例、误差水平。
   若草稿没有提供，只能写成：
   “补充实际使用的求解器/算法/随机种子/参数设置及其依据”，
   不得写成“例如使用Gurobi、种子42、gap=0%”之类看似真实的具体事实。
6. 可以给出通用的数学结构、统计量名称或符号公式；
   若使用具体数值只是教学示例，必须明确标注“示意值/需由作者根据实际数据确定”，
   且尽量优先使用符号占位而不是随意数字。
7. 对下标变量要遵循“符号族”原则：
   若草稿已一般性定义 p_i、D_i、x_i 表示第 i 类对象的对应量，
   则 p_j、D_j、x_j 通常只是同一符号族换了索引，
   不能因此声称“p_j 本身未定义”或要求逐个重复列入符号表。
   只有当 i、j 的索引集合、角色、方向关系或取值范围确实不清楚时，
   才应把问题收窄为“下标语义/索引集合说明不足”。

经验库中的数字、阈值和具体模型只属于来源案例，
不得机械要求用户照抄。

==================================================
Rubric 与 Experience 的使用方式
==================================================

Rubric 用来解释：
“这个问题属于论文质量的哪个维度”。

Experience Library 用来解释：
“优秀论文中可迁移的通用做法是什么”。

原始 RAG 案例用来解释：
“真实优秀论文具体是怎么写 / 怎么做的”。

三者不要混为一谈。

若某 issue 没有直接匹配的 experience，
不要强行制造经验。

==================================================
最终报告结构
==================================================

只生成前三部分：

# 一、总体评价

说明：
- 论文当前最主要的优点；
- 最需要优先改进的短板；
- 不给数值分数。

# 二、最终独立问题

每个 issue 必须严格按 issue_id 展开：

## Ixx：问题名（importance / priority）

### 对应 Rubric
写：
Rxx + Rubric 名称

用 1~2 句话说明为什么这个问题属于该维度。
明确说明这是内部评审框架，不需要每条重复“非官方”。

### 问题是什么

### 为什么重要

### 草稿中的核验证据

只依据用户论文 evidence / known_facts。

### 优秀经验库怎么做

优先使用提供的 Experience Library pattern：
- 说明可迁移的通用做法；
- 说明适用边界；
- 不照搬题目专属数字；
- 若 support_count 较高，可以说明这是多篇优秀论文中重复出现的做法。

如果该 issue 的 experience_guidance 为空，
必须明确写：
“Experience Library V1 暂无与该问题直接匹配的经验，本项不强行类比。”
然后直接进入优秀论文真实案例与修改建议，
不得从其他 Rubric 或弱相关经验中自行补一条。

### 优秀论文真实案例

必须使用提供的真实案例。
写明：
- paper_id
- section_title

如果 Experience Library representative source
和原始 RAG 案例都提供了内容，
优先选择与当前问题最直接的 1~2 个案例，
不要堆砌。

### 作者如何修改

给出可执行步骤。
目标是指导作者自己修改，不是代写整篇论文。

建议可包含：
- 应补充的数学结构；
- 应增加的统计量；
- 应说明的参数来源；
- 应增加的验证方式；
- 应如何组织结果分析。

但建议中的公式优先使用符号形式，
不得虚构用户论文具体数据。

修改建议的措辞规则：
- 草稿已经明确使用某工具/参数：可以针对它给具体改进建议；
- 草稿没有明确：写“补充你实际使用的……”；
- 不得替作者指定一个并未使用的工具、随机种子或运行结果；
- 不得用任意示例数字伪装成可直接写入论文的数值；
- 若建议做敏感性/收敛性实验，应建议“比较若干合理候选值/样本量”，
  具体候选值由作者结合数据规模、计算成本或预实验确定。

# 三、分模块修改建议

只讨论当前最终 issue 真正涉及的模块。

每个模块显式写：
对应 issue：Ixx

可以按：
摘要 / 问题分析 / 数据处理 / 模型建立 /
模型求解 / 结果分析 / 模型检验 / 模型评价与结论
组织。

不要生成第四部分。
第四部分由程序根据 importance 自动生成。
"""


def build_locked_reference_text_v3(
    enriched_data,
):
    blocks = []

    for issue in enriched_data.get(
        "issues",
        [],
    ):
        lock = issue.get(
            "fact_lock",
            {},
        )

        rubric = issue.get(
            "rubric",
            {},
        )

        blocks.extend([
            (
                f"## {issue['issue_id']} "
                f"{issue['problem']}"
            ),
            (
                "importance: "
                f"{issue['importance']}"
            ),
            (
                "priority: "
                f"{issue['priority']}"
            ),
            (
                "issue_kind: "
                f"{issue.get('issue_kind')}"
            ),
            (
                "rubric_id: "
                f"{rubric.get('rubric_id')}"
            ),
            (
                "rubric_name: "
                f"{rubric.get('rubric_name')}"
            ),
            (
                "rubric_reason: "
                f"{rubric.get('reason', '')}"
            ),
            (
                "reason: "
                + issue.get(
                    "reason",
                    "",
                )
            ),
            (
                "evidence_from_draft: "
                + issue.get(
                    "evidence_from_draft",
                    "",
                )
            ),
            (
                "source_sections: "
                + ", ".join(
                    issue.get(
                        "source_sections",
                        [],
                    )
                )
            ),
            (
                "known_facts: "
                + json.dumps(
                    lock.get(
                        "known_facts",
                        [],
                    ),
                    ensure_ascii=False,
                )
            ),
            (
                "removed_claims_DO_NOT_USE: "
                + json.dumps(
                    lock.get(
                        "removed_claims",
                        [],
                    ),
                    ensure_ascii=False,
                )
            ),
            (
                "forbidden_phrases_DO_NOT_USE: "
                + json.dumps(
                    lock.get(
                        "forbidden_phrases",
                        [],
                    ),
                    ensure_ascii=False,
                )
            ),
            (
                "suggested_direction: "
                + issue.get(
                    "suggested_direction",
                    "",
                )
            ),
        ])

        experiences = issue.get(
            "experience_guidance",
            [],
        )

        if experiences:
            blocks.append(
                "\n[Experience Library V1："
                "以下属于优秀论文抽象经验，不是用户论文事实]"
            )

        for exp in experiences:
            blocks.extend([
                (
                    "experience_id: "
                    f"{exp.get('experience_id')}"
                ),
                (
                    "experience_pattern: "
                    f"{exp.get('pattern')}"
                ),
                (
                    "why_effective: "
                    f"{exp.get('why_effective')}"
                ),
                (
                    "applicable_when: "
                    f"{exp.get('applicable_when')}"
                ),
                (
                    "recommended_structure: "
                    + json.dumps(
                        exp.get(
                            "recommended_structure",
                            [],
                        ),
                        ensure_ascii=False,
                    )
                ),
                (
                    "do_not_overgeneralize: "
                    f"{exp.get('do_not_overgeneralize')}"
                ),
                (
                    "experience_support: "
                    f"support_count={exp.get('support_count')}, "
                    f"evidence_strength={exp.get('evidence_strength')}"
                ),
            ])

            for source in exp.get(
                "representative_sources",
                [],
            )[:1]:
                blocks.append(
                    "Experience source "
                    "(优秀论文案例，不是用户论文事实):\n"
                    f"paper_id: {source.get('paper_id')}\n"
                    f"section_title: {source.get('title')}\n"
                    f"content:\n{source.get('excerpt')}\n"
                )

        examples = (
            issue.get(
                "retrieval",
                {}
            ).get(
                "examples",
                [],
            )
        )

        if not examples:
            blocks.append(
                "Original RAG examples: none"
            )
            continue

        blocks.append(
            "\n[原始优秀论文 RAG 案例："
            "以下属于外部优秀论文，不是用户论文事实]"
        )

        for j, ex in enumerate(
            examples,
            start=1,
        ):
            blocks.append(
                f"\nRAG Example {j}\n"
                f"relevance: "
                f"{ex.get('relevance')}\n"
                f"paper_id: "
                f"{ex.get('paper_id')}\n"
                f"section_title: "
                f"{ex.get('section_title')}\n"
                f"rerank_reason: "
                f"{ex.get('reason')}\n"
                f"content:\n"
                f"{ex.get('content')}\n"
            )

    return "\n".join(
        blocks
    )


def activate_v3_report_prompt():
    # Reuse V2.5.1's mature:
    # - long-report call
    # - factual claim audit
    # - repair loop
    # - P0/P1/P2 section
    #
    # Only replace what information is shown to the report model
    # and how the report should organize it.
    v251.FINAL_REPORT_SYSTEM = (
        V3_FINAL_REPORT_SYSTEM
    )

    v251.build_locked_reference_text = (
        build_locked_reference_text_v3
    )

    v251.REPORT_AUDIT_SYSTEM = (
        v251.REPORT_AUDIT_SYSTEM
        + "\n\n"
        + (
            "补充规则：Rubric、Experience Library、"
            "优秀论文RAG案例属于外部评审依据。"
            "只要报告明确标记其为优秀论文经验/案例，"
            "不得把其中的数字误判成用户论文事实。\n"
            "同时必须审计【修改建议本身】是否伪造具体执行事实："
            "如果草稿没有给出实际软件/库/求解器/随机种子/运行时间/"
            "optimality gap/参数值/阈值，而报告在“作者如何修改”中"
            "直接指定了诸如Gurobi、PuLP、种子42、0.5秒、gap=0%、"
            "alpha=0.6等具体值，应判为 violation。"
            "正确写法应为“补充实际使用的……及其依据”，"
            "或使用明确标注为示意的符号形式。\n"
            "另外，若 p_i 一般性表示第i类对象对应量，p_j 仅为同一"
            "符号族更换索引，不得以'p_j未单独定义'作为事实性缺陷；"
            "只能在正文确实缺少 i/j 角色、集合、范围或方向定义时，"
            "批评下标语义不清。"
        )
    )


# ============================================================
# True standalone V3 main
# ============================================================

def main():
    api_key = (
        v21.base.require_api_key()
    )

    if len(sys.argv) < 2:
        print(
            "用法：\n"
            "python review_agent_main.py "
            "\"你的论文.md\""
        )
        sys.exit(1)

    paper_path, text = (
        v21.base.read_paper(
            sys.argv[1]
        )
    )

    (
        rubric,
        experiences,
        knowledge_signature,
    ) = load_v3_knowledge()

    print("=" * 72)
    print(
        "数学建模整篇论文评审 Agent V3"
    )
    print("=" * 72)
    print(
        "论文：",
        paper_path,
    )
    print(
        "总字符数：",
        len(text),
    )
    print(
        "V3 Layer：",
        V3_LAYER_VERSION,
    )
    print(
        "Rubric：",
        RUBRIC_V1_PATH.name,
    )
    print(
        "Experience Library：",
        EXPERIENCE_V1_PATH.name,
        f"({len(experiences)} 条)",
    )
    print(
        "Rubric/Experience模型：",
        RUBRIC_EXPERIENCE_MODEL,
    )
    print(
        "Fact Audit模型：",
        v251.FACT_AUDIT_MODEL,
    )
    print(
        "Report模型：",
        v251.REPORT_MODEL,
    )

    work_dir = (
        v21.make_work_dir(
            paper_path,
            text,
        )
    )

    sections = (
        v21.split_sections_v21(
            text
        )
    )

    if not sections:
        raise RuntimeError(
            "没有识别到任何章节。"
        )

    full_outline = (
        v21.base.build_outline(
            sections
        )
    )

    clean_outline = (
        v24.build_clean_outline(
            sections
        )
    )

    outline_path = (
        work_dir
        / "01_outline.json"
    )

    _save_json_file(
        outline_path,
        {
            "paper":
                str(paper_path),
            "sections":
                sections,
            "full_outline":
                full_outline,
            "clean_outline":
                clean_outline,
        },
    )

    client = (
        v21.base.make_client(
            api_key
        )
    )

    # --------------------------------------------------------
    # 1. Local reviews, with resume checkpoint
    # --------------------------------------------------------
    progress_path = (
        work_dir
        / "02_section_reviews_progress.json"
    )

    local_reviews, local_issues = (
        v21.review_all_sections_resume(
            client=client,
            sections=sections,
            outline_text=
                full_outline,
            checkpoint_path=
                progress_path,
        )
    )

    section_reviews_path = (
        work_dir
        / "02_section_reviews.json"
    )

    _save_json_file(
        section_reviews_path,
        local_reviews,
    )

    # --------------------------------------------------------
    # 2. Global structure
    # --------------------------------------------------------
    global_path = (
        work_dir
        / "03_global_review.json"
    )

    global_review = (
        v21.load_json(
            global_path
        )
    )

    if global_review is None:
        global_review = (
            v21.review_global_structure_v21(
                client=client,
                sections=sections,
                outline_text=
                    full_outline,
            )
        )

        _save_json_file(
            global_path,
            global_review,
        )
    else:
        print(
            "[2/6] 检测到全文结构缓存，直接复用。"
        )

    # --------------------------------------------------------
    # 3. Initial merge
    # --------------------------------------------------------
    merged_path = (
        work_dir
        / "04_merged_issues.json"
    )

    merged_data = (
        v21.load_json(
            merged_path
        )
    )

    if merged_data is None:
        merged_data = (
            v21.merge_issues_v21(
                client=client,
                local_issues=
                    local_issues,
                global_issues=
                    global_review.get(
                        "issues",
                        [],
                    ),
            )
        )

        _save_json_file(
            merged_path,
            merged_data,
        )
    else:
        print(
            "[3.0/6] 检测到初步合并缓存，直接复用。"
        )

    if not merged_data.get(
        "issues"
    ):
        print(
            "未识别到需要继续核验的高价值问题。"
        )
        return

    # --------------------------------------------------------
    # 3.5 V2.4 Evidence Calibration
    # --------------------------------------------------------
    verified_path = (
        work_dir
        / "04b_verified_issues_v24.json"
    )

    verified_data = (
        v24.load_json(
            verified_path
        )
    )

    if verified_data is None:
        verified_data = (
            v24.verify_all_v24(
                client=client,
                merged_data=
                    merged_data,
                sections=sections,
                outline=
                    clean_outline,
            )
        )

        _save_json_file(
            verified_path,
            verified_data,
        )
    else:
        print(
            "[3.5/6] 检测到 V2.4 Evidence "
            "Calibration 缓存，直接复用。"
        )

    if not verified_data.get(
        "issues"
    ):
        print(
            "Evidence Calibration 后没有问题保留。"
        )
        return

    # --------------------------------------------------------
    # 3.7 Post-Evidence Consolidation
    # --------------------------------------------------------
    consolidated_path = (
        work_dir
        / "04c_consolidated_issues_v25.json"
    )

    consolidated_data = (
        v25.load_json(
            consolidated_path
        )
    )

    if consolidated_data is None:
        consolidated_data = (
            v25.consolidate_verified_issues(
                client=client,
                verified_data=
                    verified_data,
            )
        )

        _save_json_file(
            consolidated_path,
            consolidated_data,
        )
    else:
        print(
            "[3.7/6] 检测到 V2.5 consolidation "
            "缓存，直接复用。"
        )

    if not consolidated_data.get(
        "issues"
    ):
        print(
            "Post-Evidence Consolidation "
            "后没有问题保留。"
        )
        return

    # --------------------------------------------------------
    # 3.8 Cross-section Fact Audit
    # --------------------------------------------------------
    fact_audit_path = (
        work_dir
        / "04d_cross_section_fact_audit_v251.json"
    )

    audited_data = (
        v251.load_json(
            fact_audit_path
        )
    )

    if audited_data is None:
        audited_data = (
            v251.audit_all_issues(
                client=client,
                consolidated_data=
                    consolidated_data,
                sections=sections,
                outline=
                    clean_outline,
            )
        )

        _save_json_file(
            fact_audit_path,
            audited_data,
        )
    else:
        print(
            "[3.8/6] 检测到 Cross-section Fact Audit "
            "缓存，直接复用。"
        )

    if not audited_data.get(
        "issues"
    ):
        print(
            "Cross-section Fact Audit "
            "后没有问题保留。"
        )
        return

    # --------------------------------------------------------
    # 3.9 Final Overlap Audit
    # --------------------------------------------------------
    final_issue_path = (
        work_dir
        / "04e_final_issues_v251.json"
    )

    final_issue_data = (
        v251.load_json(
            final_issue_path
        )
    )

    if final_issue_data is None:
        final_issue_data = (
            v251.overlap_audit(
                client=client,
                audited_data=
                    audited_data,
            )
        )

        _save_json_file(
            final_issue_path,
            final_issue_data,
        )
    else:
        print(
            "[3.9/6] 检测到 Final Overlap Audit "
            "缓存，直接复用。"
        )

    if not final_issue_data.get(
        "issues"
    ):
        print(
            "最终没有问题保留，停止。"
        )
        return

    # Signature guarantees V3 cache is invalidated
    # if final issues or frozen knowledge change.
    v3_signature = hashlib.sha256(
        (
            V3_LAYER_VERSION
            + "-"
            + knowledge_signature
            + "-"
            + _issue_sha12(
                final_issue_data
            )
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    # --------------------------------------------------------
    # 4.0 Rubric + Experience
    # --------------------------------------------------------
    rubric_exp_path = (
        work_dir
        / "04f_rubric_experience_v3.json"
    )

    rubric_exp_data = (
        _load_cache_with_signature(
            rubric_exp_path,
            v3_signature,
        )
    )

    if rubric_exp_data is None:
        rubric_exp_data = (
            attach_rubric_and_experience(
                client=client,
                final_issue_data=
                    final_issue_data,
                rubric=rubric,
                experiences=
                    experiences,
            )
        )

        rubric_exp_data[
            "v3_signature"
        ] = v3_signature

        rubric_exp_data[
            "knowledge_signature"
        ] = knowledge_signature

        _save_json_file(
            rubric_exp_path,
            rubric_exp_data,
        )
    else:
        print(
            "[4.0/6] 检测到 Rubric + Experience "
            "缓存，直接复用。"
        )

    # --------------------------------------------------------
    # 5. Original RAG + rerank,
    #    but use Experience-enhanced query
    # --------------------------------------------------------
    retrieval_path = (
        work_dir
        / "05_retrieval_v3.json"
    )

    enriched_data = (
        _load_cache_with_signature(
            retrieval_path,
            v3_signature,
        )
    )

    if enriched_data is None:
        collection, embedding_model = (
            v21.base.initialize_retrieval()
        )

        enriched_data = (
            retrieve_v3(
                client=client,
                rubric_experience_data=
                    rubric_exp_data,
                embedding_model=
                    embedding_model,
                collection=collection,
            )
        )

        enriched_data[
            "v3_signature"
        ] = v3_signature

        enriched_data[
            "rubric_framework"
        ] = rubric_exp_data.get(
            "rubric_framework",
            {},
        )

        _save_json_file(
            retrieval_path,
            enriched_data,
        )
    else:
        print(
            "[5/6] 检测到 V3 RAG 缓存，直接复用。"
        )

    # --------------------------------------------------------
    # 6. V3 report, reuse V2.5.1 factual lock/audit loop
    # --------------------------------------------------------
    activate_v3_report_prompt()

    report_path = (
        work_dir
        / "review_report_v3.md"
    )

    report_audit_path = (
        work_dir
        / "06_report_fact_audit_v3.json"
    )

    report_meta_path = (
        work_dir
        / "06_report_meta_v3.json"
    )

    can_reuse_report = False

    if (
        report_path.exists()
        and report_meta_path.exists()
    ):
        try:
            meta = _load_json_file(
                report_meta_path
            )

            report = (
                report_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )

            if (
                meta.get(
                    "v3_signature"
                )
                == v3_signature
                and v25.report_complete(
                    report,
                    enriched_data.get(
                        "issues",
                        [],
                    ),
                )
            ):
                can_reuse_report = True
        except Exception:
            can_reuse_report = False

    if can_reuse_report:
        print(
            "[6/6] 检测到完整 V3 报告，直接复用。"
        )
    else:
        print(
            "\n[6/6] 正在生成 Agent V3 最终报告..."
        )

        report, report_audit = (
            v251.generate_final_report(
                client=client,
                outline=
                    clean_outline,
                enriched_data=
                    enriched_data,
            )
        )

        report_path.write_text(
            report,
            encoding="utf-8",
        )

        _save_json_file(
            report_audit_path,
            report_audit,
        )

        _save_json_file(
            report_meta_path,
            {
                "v3_signature":
                    v3_signature,
                "knowledge_signature":
                    knowledge_signature,
                "rubric_path":
                    str(
                        RUBRIC_V1_PATH
                    ),
                "experience_path":
                    str(
                        EXPERIENCE_V1_PATH
                    ),
                "experience_count":
                    len(
                        experiences
                    ),
            },
        )

    print()
    print("=" * 72)
    print(
        "Agent V3 最终报告"
    )
    print("=" * 72)
    print()
    print(report)

    print()
    print("=" * 72)
    print(
        "V3 输出文件"
    )
    print("=" * 72)
    print(
        "Rubric + Experience：",
        rubric_exp_path,
    )
    print(
        "RAG：",
        retrieval_path,
    )
    print(
        "Report Fact Audit：",
        report_audit_path,
    )
    print(
        "最终报告：",
        report_path,
    )


if __name__ == "__main__":
    main()
