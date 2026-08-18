import json
import re
from collections import Counter
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")

CN_DIGITS = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = (
        text.replace("⼀", "一")
            .replace("⼆", "二")
            .replace("⼈", "人")
            .replace("（", "(")
            .replace("）", ")")
            .replace("：", ":")
            .replace("．", ".")
    )
    return text


def compact_title(title: str) -> str:
    title = normalize_text(title).lower()
    return re.sub(r"\s+", "", title)


def chinese_number_to_int(s: str):
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in CN_DIGITS:
        return CN_DIGITS[s]
    if "十" in s:
        if s == "十":
            return 10
        left, _, right = s.partition("十")
        tens = CN_DIGITS.get(left, 1) if left else 1
        ones = CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def extract_question_number(title: str):
    t = compact_title(title)

    for pattern in [
        r"问题([一二三四五六七八九十百\d]+)",
        r"第([一二三四五六七八九十百\d]+)问",
    ]:
        m = re.search(pattern, t)
        if m:
            return chinese_number_to_int(m.group(1))

    return None


def is_major_heading(title: str) -> bool:
    t = normalize_text(title).strip()

    if re.match(r"^[一二三四五六七八九十]+[、.]", t):
        return True

    # 1. xxx / 2. xxx；排除 4.1 / 5.2.3
    if re.match(r"^\d+\.(?!\d)", t):
        return True

    return False


def extract_major_number(title: str):
    t = normalize_text(title).strip()

    m = re.match(r"^([一二三四五六七八九十]+)[、.]", t)
    if m:
        return chinese_number_to_int(m.group(1))

    m = re.match(r"^(\d+)\.(?!\d)", t)
    if m:
        return int(m.group(1))

    return None


def extract_numeric_path(title: str):
    t = normalize_text(title).strip()
    m = re.match(r"^(\d+(?:\.\d+)+)", t)

    if not m:
        return None

    try:
        return [int(x) for x in m.group(1).split(".")]
    except ValueError:
        return None


def looks_like_code_heading(title: str) -> bool:
    t = compact_title(title)

    patterns = [
        r"^todo[:：]?",
        r"^print\(",
        r"^for\b",
        r"^if\b",
        r"^while\b",
        r"^np\.",
        r"^pd\.",
        r"^[a-z_][a-z0-9_]*=",
        r"^[a-z_][a-z0-9_]*\+=",
    ]

    return any(re.search(p, t) for p in patterns)


def classify_direct(title: str, current_context=None):
    """
    先根据“当前标题自身”判断语义。
    只有当前标题没有明显语义时，才交给父级上下文继承。
    """
    t = compact_title(title)

    # =========================================================
    # 附录 / 代码 / 参考文献
    # =========================================================

    if (
        "附录" in t
        or "支撑材料目录" in t
        or "代码汇总" in t
        or "源代码" in t
        or "程序代码" in t
        or "ai使用记录" in t
    ):
        return "appendix"

    if current_context in {"appendix", "reference"}:
        if (
            looks_like_code_heading(title)
            or re.match(
                r"^(程序[一二三四五六七八九十\d]|"
                r"题[一二三四五六七八九十\d]+代码|代码)",
                t,
            )
        ):
            return "appendix"

    if "参考文献" in t or t in {
        "reference", "references", "文献"
    }:
        return "reference"

    # =========================================================
    # 前置模块
    # =========================================================

    if t in {"摘要", "abstract"}:
        return "abstract"

    if any(k in t for k in [
        "问题背景", "研究背景", "背景介绍"
    ]):
        return "background"

    if any(k in t for k in [
        "问题重述", "题目重述", "问题要求",
        "需要解决的问题", "基本问题", "问题描述"
    ]):
        return "problem_statement"

    if any(k in t for k in [
        "模型假设", "基本假设", "问题假设", "建模假设"
    ]):
        return "assumption"

    if any(k in t for k in [
        "符号说明", "符号定义", "符号约定", "符号表"
    ]):
        return "notation"

    # =========================================================
    # 检验 / 灵敏度 / 评价
    # 必须在普通“分析”前判断
    # =========================================================

    if any(k in t for k in [
        "误差分析", "误差估计", "误差检验"
    ]):
        return "error_analysis"

    if any(k in t for k in [
        "灵敏度", "敏感性"
    ]):
        return "sensitivity"

    if any(k in t for k in [
        "鲁棒性",
        "模型检验", "模型验证", "模型校验",
        "结果验证", "结果检验",
        "可靠性检验", "稳定性检验",
        "显著性检验", "拟合优度", "残差分析",
        "模型评价和检验", "模型评价与检验"
    ]):
        return "validation"

    if (
        ("优点" in t and ("缺点" in t or "不足" in t))
        or ("评价" in t and "推广" in t)
    ):
        return "evaluation"

    if any(k in t for k in [
        "模型的优点", "模型优点",
        "模型优势", "模型的优势", "优点分析"
    ]):
        return "strength"

    if any(k in t for k in [
        "模型的缺点", "模型缺点",
        "模型不足", "模型的不足",
        "模型局限", "局限性"
    ]):
        return "weakness"

    if any(k in t for k in [
        "模型推广", "模型的推广",
        "推广应用", "应用前景", "展望"
    ]):
        return "generalization"

    if any(k in t for k in [
        "模型评价", "模型评估",
        "优劣性分析", "模型优劣"
    ]):
        return "evaluation"

    # =========================================================
    # 数据处理
    # =========================================================

    if any(k in t for k in [
        "数据预处理", "数据初步处理",
        "数据清洗", "缺失值处理", "异常值处理",
        "数据处理及", "数据处理与", "数据处理"
    ]):
        return "preprocessing"

    # =========================================================
    # 建模 / 求解
    # =========================================================

    if any(k in t for k in [
        "模型的比较与选择", "模型比较与选择",
        "模型选择", "模型的选择", "方法选择"
    ]):
        return "model_selection"

    if any(k in t for k in [
        "建模思路", "模型思路",
        "求解思路", "总体思路", "模型构思"
    ]):
        return "model_idea"

    # “建模与求解”只作为父容器，避免所有子标题都变成 solving
    if (
        ("建模" in t and "求解" in t)
        or (
            "模型" in t
            and "建立" in t
            and "求解" in t
        )
    ):
        return "modeling_container"

    if any(k in t for k in [
        "求解结果", "计算结果", "结果展示",
        "结果分析", "结果汇总",
        "结果总结", "求解结果与分析"
    ]):
        return "result"

    if "小结" in t or "结论" in t:
        return "result"

    if any(k in t for k in [
        "模型求解", "模型的求解",
        "求解模型", "优化模型求解",
        "求解步骤", "算法介绍", "算法设计",
        "迭代求解", "循环迭代",
        "二分法", "二分查找", "搜索寻优",
        "遗传算法", "差分进化算法",
        "模拟退火", "蒙特卡洛",
        "模型的应用及求解", "模型应用及求解"
    ]):
        return "model_solving"

    if re.search(
        r"(模型.*(建立|构建|设定|结构|表达式|汇总)|"
        r"建立.*模型|构建.*模型)",
        t,
    ):
        return "model_building"

    if any(k in t for k in [
        "决策变量", "状态变量",
        "目标函数", "约束条件",
        "成本函数", "收益函数",
        "轨迹方程", "判定式",
        "坐标系的建立", "模型准备"
    ]):
        return "model_building"

    # Bayesian模型、CVaR模型等标题
    if (
        "模型" in t
        and "分析" not in t
        and "检验" not in t
        and "验证" not in t
    ):
        return "model_building"

    # =========================================================
    # 普通分析最后判断
    # =========================================================

    if any(k in t for k in [
        "问题分析", "相关性分析",
        "可视化分析", "数据分析", "机理分析"
    ]):
        return "analysis"

    if t.endswith("分析"):
        return "analysis"

    return None


MAIN_SECTION_MAP = {
    "abstract": "abstract",
    "background": "problem",
    "problem_statement": "problem",
    "analysis": "analysis",
    "assumption": "assumption",
    "notation": "notation",
    "preprocessing": "preprocessing",

    "modeling": "modeling",
    "model_idea": "modeling",
    "model_selection": "modeling",
    "model_building": "modeling",
    "model_solving": "modeling",
    "model_detail": "modeling",
    "result": "modeling",

    "validation": "validation",
    "error_analysis": "validation",
    "sensitivity": "validation",

    "evaluation": "evaluation",
    "strength": "evaluation",
    "weakness": "evaluation",
    "generalization": "evaluation",

    "reference": "reference",
    "appendix": "appendix",
    "other": "other",
}


def infer_from_context(current_context):
    if current_context in {
        "modeling",
        "model_idea",
        "model_selection",
        "model_building",
        "model_solving",
    }:
        return "model_detail"

    if current_context in {
        "abstract",
        "background",
        "problem_statement",
        "analysis",
        "assumption",
        "notation",
        "preprocessing",
        "result",
        "validation",
        "error_analysis",
        "sensitivity",
        "evaluation",
        "strength",
        "weakness",
        "generalization",
        "reference",
        "appendix",
    }:
        return current_context

    return "other"


def split_markdown(md_text: str):
    sections = []

    current_title = None
    current_level = None
    buffer = []

    def flush():
        nonlocal current_title, current_level, buffer

        if current_title is None:
            buffer = []
            return

        sections.append({
            "title": normalize_text(current_title),
            "markdown_level": current_level,
            "content": "\n".join(buffer).strip(),
        })

        buffer = []

    for line in md_text.splitlines():
        m = HEADING_RE.match(line)

        if m:
            flush()
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
        else:
            if current_title is not None:
                buffer.append(line)

    flush()
    return sections


def process_paper(
    input_path,
    output_path,
    paper_id,
    problem_id,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    text = input_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    raw_sections = split_markdown(text)

    results = []

    current_context = None
    current_question = None

    # 适配：
    # “五、模型建立与求解”
    # “5.1 问题一 / 5.2 问题二 ...”
    modeling_group_chapter = None

    for raw in raw_sections:
        title = raw["title"]
        content = raw["content"]

        explicit_question = extract_question_number(title)

        direct_type = classify_direct(
            title,
            current_context=current_context,
        )

        major = is_major_heading(title)
        major_number = extract_major_number(title)

        if explicit_question is not None:
            current_question = explicit_question

        if direct_type == "modeling_container":
            section_type = "model_detail"
            new_context = "modeling"

            if major and explicit_question is None:
                modeling_group_chapter = major_number

        elif direct_type is not None:
            section_type = direct_type
            new_context = direct_type

            if major:
                if direct_type not in {
                    "model_building",
                    "model_solving",
                    "model_detail",
                }:
                    modeling_group_chapter = None

        else:
            section_type = infer_from_context(
                current_context
            )
            new_context = current_context

        # 兜底推断问题编号：
        # 某章统一写“模型建立与求解”时，5.1/5.2可对应问题1/2
        if explicit_question is None:
            numeric_path = extract_numeric_path(title)

            if (
                current_question is None
                and modeling_group_chapter is not None
                and numeric_path is not None
                and len(numeric_path) >= 2
                and numeric_path[0] == modeling_group_chapter
                and 1 <= numeric_path[1] <= 20
            ):
                current_question = numeric_path[1]

        # 新的全局大模块，不应继承上一问的问题编号
        if (
            major
            and explicit_question is None
            and direct_type in {
                "abstract",
                "background",
                "problem_statement",
                "analysis",
                "assumption",
                "notation",
                "preprocessing",
                "validation",
                "error_analysis",
                "sensitivity",
                "evaluation",
                "strength",
                "weakness",
                "generalization",
                "reference",
                "appendix",
            }
        ):
            current_question = None

        if section_type in {
            "reference",
            "appendix",
        }:
            current_question = None

        main_section = MAIN_SECTION_MAP.get(
            section_type,
            "other",
        )

        # 最前面的论文题目如果没有正文，不进入 section KB
        if (
            not content
            and not results
            and section_type == "other"
        ):
            current_context = new_context
            continue

        section_id = (
            f"{paper_id}_S"
            f"{len(results) + 1:03d}"
        )

        results.append({
            "section_id": section_id,
            "paper_id": paper_id,
            "problem_id": problem_id,
            "main_section": main_section,
            "section_type": section_type,
            "section_title": title,
            "question_number": current_question,
            "content": content,
        })

        if new_context is not None:
            current_context = new_context

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in results:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    type_count = Counter(
        x["section_type"]
        for x in results
    )

    q_count = Counter(
        x["question_number"]
        for x in results
    )

    print(
        f"处理完成，共 {len(results)} 个章节"
    )

    print("\n章节类型：")
    for key, value in type_count.items():
        print(f"{key}: {value}")

    print("\n问题编号：")
    for key, value in q_count.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    print(
        "请通过 build_all_structured.py "
        "批量调用 process_paper()。"
    )
