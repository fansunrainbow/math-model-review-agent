from collections import defaultdict
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path("D:/math_model_kb")
DB_PATH = ROOT / "database" / "chroma"
MODEL_NAME = str(ROOT / "models" / "bge-base-zh-v1.5")
COLLECTION_NAME = "math_model_papers_all_v1"

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 每次先多取一些候选，再做质量过滤 + 多样性重排
DEFAULT_CANDIDATE_K = 80

# 同一篇论文最多返回几条
MAX_PER_PAPER = 1

# 最终返回几条
DEFAULT_TOP_K = 5


# ============================================================
# 1. 查询意图识别
# ============================================================

def detect_section_types(query: str):
    q = query.replace(" ", "")

    if "灵敏度分析" in q or "敏感性分析" in q:
        return ["sensitivity"]

    if "模型假设" in q or "建模假设" in q:
        return ["assumption"]

    if (
        "优缺点" in q
        or "优点和缺点" in q
        or "优点与缺点" in q
        or "优势和不足" in q
    ):
        return ["strength", "weakness"]

    if "模型优点" in q or "模型优势" in q:
        return ["strength"]

    if (
        "模型不足" in q
        or "模型缺点" in q
        or "模型局限" in q
        or "局限性" in q
    ):
        return ["weakness"]

    if "误差分析" in q:
        return ["error_analysis"]

    if (
        "模型检验" in q
        or "模型验证" in q
        or "结果验证" in q
        or "结果检验" in q
        or "鲁棒性" in q
    ):
        return ["validation"]

    if "模型推广" in q or "推广应用" in q:
        return ["generalization"]

    if "数据预处理" in q or "数据处理" in q:
        return ["preprocessing"]

    if "模型选择" in q or "模型比较" in q:
        return ["model_selection", "model_idea"]

    if (
        "模型建立" in q
        or "模型构建" in q
        or "目标函数" in q
        or "约束条件" in q
        or "决策变量" in q
    ):
        return ["model_building", "model_detail"]

    if (
        "模型求解" in q
        or "求解算法" in q
        or "怎么求解" in q
        or "如何求解" in q
    ):
        return ["model_solving", "model_detail"]

    return None


def build_where(section_types=None):
    if not section_types:
        return None

    if len(section_types) == 1:
        return {
            "section_type": section_types[0]
        }

    return {
        "section_type": {
            "$in": section_types
        }
    }


# ============================================================
# 2. 文档正文提取
#    build_vector_db 时前面拼了元数据，这里把真正正文取出来
# ============================================================

def extract_body(document: str):
    if not document:
        return ""

    # 我们 build_search_text 里在“章节标题”后加了一个空行，再接正文
    parts = document.split("\n\n", 1)

    if len(parts) == 2:
        return parts[1].strip()

    return document.strip()


# ============================================================
# 3. 低质量 / 代码污染检测
# ============================================================

CODE_PATTERNS = [
    r"\bparser\.add_argument\s*\(",
    r"\bimport\s+[a-zA-Z_]",
    r"\bfrom\s+[a-zA-Z_][\w.]*\s+import\b",
    r"\bdef\s+[a-zA-Z_]\w*\s*\(",
    r"\bclass\s+[a-zA-Z_]\w*",
    r"\bprint\s*\(",
    r"\bfor\s+\w+\s+in\s+",
    r"\bwhile\s+",
    r"\bif\s+.+:",
    r"\bnp\.",
    r"\bpd\.",
    r"\bplt\.",
    r"\bsklearn\b",
    r"\bmodel\.fit\s*\(",
    r"\bmodel\.predict\s*\(",
    r"\bargparse\b",
    r"\blambda\s+",
    r"^\s*[a-zA-Z_]\w*\s*=",
]

LOW_INFO_PHRASES = [
    "模型假设主要包括",
    "结果如下",
    "具体如下",
    "如下所示",
    "见下图",
    "见下表",
    "计算结果如下",
]


def code_score(text: str):
    """
    返回一个简单的代码污染分数。
    分数越高越像代码。
    """
    if not text:
        return 0

    score = 0

    for pattern in CODE_PATTERNS:
        if re.search(
            pattern,
            text,
            flags=re.I | re.M
        ):
            score += 1

    # 特殊字符密度
    special_chars = sum(
        text.count(ch)
        for ch in [
            "=", "{", "}", "[", "]",
            "(", ")", ";"
        ]
    )

    if len(text) > 0:
        ratio = special_chars / len(text)

        if ratio > 0.08:
            score += 1

        if ratio > 0.15:
            score += 1

    # 多行明显像代码
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    code_like_lines = 0

    for line in lines:
        if (
            re.match(
                r"^(import |from |def |class |for |while |if |elif |else:|return |print\(|#)",
                line
            )
            or "=" in line and len(line) < 180
        ):
            code_like_lines += 1

    if len(lines) >= 3:
        if code_like_lines / len(lines) >= 0.5:
            score += 2

    return score


def is_low_quality_result(row):
    meta = row["metadata"]
    body = extract_body(row["document"])
    title = (
        meta.get("section_title")
        or ""
    ).strip()

    # 1. 正文太短
    # 注意：某些 weakness/strength 本来就短，因此阈值不要太高
    if len(body) < 18:
        return True

    # 2. 极低信息句
    normalized = re.sub(
        r"\s+",
        "",
        body
    )

    for phrase in LOW_INFO_PHRASES:
        if normalized == re.sub(
            r"\s+",
            "",
            phrase
        ):
            return True

    # 3. 正文几乎只有表格，而且标题本身不是表格相关内容
    if (
        body.startswith("<table")
        and body.endswith("</table>")
        and len(
            re.sub(
                r"<[^>]+>",
                "",
                body
            ).strip()
        ) < 60
    ):
        return True

    # 4. 明显代码污染
    if code_score(body) >= 3:
        return True

    # 5. 标题本身明显就是代码
    if code_score(title) >= 2:
        return True

    return False


# ============================================================
# 4. Chroma 查询
# ============================================================

def query_candidates(
    model,
    collection,
    query,
    section_types=None,
    candidate_k=DEFAULT_CANDIDATE_K,
):
    where = build_where(
        section_types
    )

    query_text = (
        QUERY_INSTRUCTION
        + query
    )

    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    kwargs = {
        "query_embeddings": [
            embedding.tolist()
        ],
        "n_results": candidate_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:
        kwargs["where"] = where

    result = collection.query(
        **kwargs
    )

    rows = []

    for (
        chunk_id,
        document,
        metadata,
        distance
    ) in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append({
            "chunk_id": chunk_id,
            "document": document,
            "metadata": metadata,
            "distance": float(distance),
        })

    return rows, where


# ============================================================
# 5. 结果重排
# ============================================================

def filter_quality(rows):
    good = []
    bad = []

    for row in rows:
        if is_low_quality_result(row):
            bad.append(row)
        else:
            good.append(row)

    return good, bad


def diversify_single_type(
    rows,
    top_k=DEFAULT_TOP_K,
    max_per_paper=MAX_PER_PAPER,
):
    selected = []
    paper_count = defaultdict(int)
    seen_content = set()

    for row in rows:
        paper_id = row["metadata"].get(
            "paper_id",
            "UNKNOWN"
        )

        if (
            paper_count[paper_id]
            >= max_per_paper
        ):
            continue

        # 简单正文去重
        body = extract_body(
            row["document"]
        )

        fingerprint = re.sub(
            r"\s+",
            "",
            body
        )[:500]

        if fingerprint in seen_content:
            continue

        seen_content.add(
            fingerprint
        )

        selected.append(
            row
        )

        paper_count[
            paper_id
        ] += 1

        if len(selected) >= top_k:
            break

    return selected


def balanced_multi_type(
    rows,
    section_types,
    top_k=DEFAULT_TOP_K,
    max_per_paper=MAX_PER_PAPER,
):
    """
    多类型查询时进行类别平衡。

    例如：
    strength + weakness
    top_k=5

    会尽量得到：
    strength 2
    weakness 2
    剩余1条按距离补
    """

    grouped = defaultdict(list)

    for row in rows:
        stype = row["metadata"].get(
            "section_type"
        )
        grouped[stype].append(
            row
        )

    selected = []
    used_ids = set()
    paper_count = defaultdict(int)

    # 每种类型至少先取 quota 条
    quota = max(
        1,
        top_k // len(section_types)
    )

    for stype in section_types:
        taken = 0

        for row in grouped.get(
            stype,
            []
        ):
            paper_id = row["metadata"].get(
                "paper_id",
                "UNKNOWN"
            )

            if (
                paper_count[paper_id]
                >= max_per_paper
            ):
                continue

            if row["chunk_id"] in used_ids:
                continue

            selected.append(
                row
            )

            used_ids.add(
                row["chunk_id"]
            )

            paper_count[
                paper_id
            ] += 1

            taken += 1

            if (
                taken >= quota
                or len(selected) >= top_k
            ):
                break

        if len(selected) >= top_k:
            return selected

    # 剩余名额按距离补
    remaining = sorted(
        rows,
        key=lambda x: x["distance"]
    )

    for row in remaining:
        if row["chunk_id"] in used_ids:
            continue

        paper_id = row["metadata"].get(
            "paper_id",
            "UNKNOWN"
        )

        if (
            paper_count[paper_id]
            >= max_per_paper
        ):
            continue

        selected.append(
            row
        )

        used_ids.add(
            row["chunk_id"]
        )

        paper_count[
            paper_id
        ] += 1

        if len(selected) >= top_k:
            break

    return selected


def rerank_results(
    rows,
    section_types,
    top_k=DEFAULT_TOP_K,
):
    good_rows, bad_rows = filter_quality(
        rows
    )

    # 查询结果本身已经按 distance 排序
    if (
        section_types
        and len(section_types) > 1
    ):
        selected = balanced_multi_type(
            good_rows,
            section_types,
            top_k=top_k,
        )
    else:
        selected = diversify_single_type(
            good_rows,
            top_k=top_k,
        )

    return selected, bad_rows


# ============================================================
# 6. 搜索接口
# ============================================================

def search(
    model,
    collection,
    query,
    top_k=DEFAULT_TOP_K,
    candidate_k=DEFAULT_CANDIDATE_K,
):
    section_types = (
        detect_section_types(
            query
        )
    )

    rows, where = (
        query_candidates(
            model=model,
            collection=collection,
            query=query,
            section_types=section_types,
            candidate_k=candidate_k,
        )
    )

    selected, rejected = (
        rerank_results(
            rows=rows,
            section_types=section_types,
            top_k=top_k,
        )
    )

    return {
        "query": query,
        "section_types": section_types,
        "where": where,
        "results": selected,
        "rejected_count": len(rejected),
        "candidate_count": len(rows),
    }


# ============================================================
# 7. 输出
# ============================================================

def print_results(search_result):
    query = search_result["query"]
    section_types = (
        search_result[
            "section_types"
        ]
    )
    where = (
        search_result["where"]
    )
    rows = (
        search_result["results"]
    )

    print()
    print("=" * 80)
    print("查询：", query)

    print(
        "识别章节类型：",
        section_types
    )

    print(
        "Chroma过滤条件：",
        where
    )

    print(
        "候选数量：",
        search_result[
            "candidate_count"
        ]
    )

    print(
        "低质量过滤：",
        search_result[
            "rejected_count"
        ]
    )

    print("=" * 80)

    if not rows:
        print(
            "没有找到足够高质量的结果。"
        )
        return

    for i, row in enumerate(
        rows,
        start=1,
    ):
        meta = row[
            "metadata"
        ]

        body = extract_body(
            row[
                "document"
            ]
        )

        print()
        print(
            f"[结果 {i}]"
        )

        print(
            "paper_id：",
            meta.get(
                "paper_id"
            )
        )

        print(
            "problem_id：",
            meta.get(
                "problem_id"
            )
        )

        if (
            meta.get(
                "question_number"
            )
            is not None
        ):
            print(
                "question_number：",
                meta.get(
                    "question_number"
                )
            )

        print(
            "section_type：",
            meta.get(
                "section_type"
            )
        )

        print(
            "section_title：",
            meta.get(
                "section_title"
            )
        )

        print(
            "distance：",
            round(
                row[
                    "distance"
                ],
                4
            )
        )

        print(
            "chunk_id：",
            row[
                "chunk_id"
            ]
        )

        print("-" * 80)

        print(
            body[:1500]
        )

        print()


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print(
        "正在连接 Chroma..."
    )

    client = (
        chromadb.PersistentClient(
            path=str(
                DB_PATH
            )
        )
    )

    collection = (
        client.get_collection(
            name=COLLECTION_NAME
        )
    )

    print(
        "Collection：",
        COLLECTION_NAME
    )

    print(
        "记录数量：",
        collection.count()
    )

    print(
        "正在加载本地 BGE 模型..."
    )

    model = (
        SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )
    )

    print(
        "BGE 模型加载完成。"
    )

    print(
        "输入 exit / quit 退出。"
    )

    while True:
        print()

        query = input(
            "请输入查询："
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        try:
            result = search(
                model=model,
                collection=collection,
                query=query,
                top_k=5,
                candidate_k=80,
            )

            print_results(
                result
            )

        except Exception as e:
            print(
                "检索失败：",
                repr(e)
            )


if __name__ == "__main__":
    main()
