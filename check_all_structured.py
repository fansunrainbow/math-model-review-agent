import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path("D:/math_model_kb")

SECTIONS_PATH = (
    ROOT
    / "processed_all"
    / "all_sections.jsonl"
)

CHUNKS_PATH = (
    ROOT
    / "processed_all"
    / "all_chunks.jsonl"
)

REPORT_PATH = (
    ROOT
    / "processed_all"
    / "structured_check.json"
)


def read_jsonl(path):
    result = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                result.append(
                    json.loads(line)
                )

    return result


def is_atomic_long_chunk(text):
    """
    长 chunk 如果主要是完整表格/公式，
    暂时认为是允许的。
    """

    text = text.strip()

    if (
        "<table" in text
        and "</table>" in text
    ):
        return True

    # 整段包含 display math
    if text.count("$$") >= 2:
        return True

    return False


def main():

    sections = read_jsonl(
        SECTIONS_PATH
    )

    chunks = read_jsonl(
        CHUNKS_PATH
    )

    print("=" * 70)
    print("开始检查...")
    print("sections：", len(sections))
    print("chunks：", len(chunks))
    print("=" * 70)

    # ==================================================
    # 1. 全局 section 类型统计
    # ==================================================

    section_type_count = Counter(
        x.get(
            "section_type",
            "MISSING"
        )
        for x in sections
    )

    # ==================================================
    # 2. 按论文统计
    # ==================================================

    paper_sections = defaultdict(list)
    paper_chunks = defaultdict(list)

    for x in sections:
        paper_sections[
            x["paper_id"]
        ].append(x)

    for x in chunks:
        paper_chunks[
            x["paper_id"]
        ].append(x)

    paper_stats = []
    suspicious_papers = []

    for paper_id in sorted(
        paper_sections
    ):

        ss = paper_sections[
            paper_id
        ]

        cc = paper_chunks.get(
            paper_id,
            []
        )

        types = Counter(
            x.get(
                "section_type",
                "MISSING"
            )
            for x in ss
        )

        total_sections = len(ss)

        other_count = types.get(
            "other",
            0
        )

        analysis_count = types.get(
            "analysis",
            0
        )

        reference_count = types.get(
            "reference",
            0
        )

        appendix_count = types.get(
            "appendix",
            0
        )

        other_ratio = (
            other_count
            / total_sections
            if total_sections
            else 0
        )

        analysis_ratio = (
            analysis_count
            / total_sections
            if total_sections
            else 0
        )

        reference_ratio = (
            reference_count
            / total_sections
            if total_sections
            else 0
        )

        appendix_ratio = (
            appendix_count
            / total_sections
            if total_sections
            else 0
        )

        has_model = any(
            types.get(t, 0) > 0
            for t in [
                "model_idea",
                "model_selection",
                "model_building",
                "model_solving",
                "model_detail"
            ]
        )

        reasons = []

        if other_ratio >= 0.30:
            reasons.append(
                f"other占比过高: "
                f"{other_count}/{total_sections}"
            )

        if analysis_ratio >= 0.50:
            reasons.append(
                f"analysis占比过高: "
                f"{analysis_count}/{total_sections}"
            )

        if reference_ratio >= 0.25:
            reasons.append(
                f"reference占比过高: "
                f"{reference_count}/{total_sections}"
            )

        if appendix_ratio >= 0.50:
            reasons.append(
                f"appendix占比过高: "
                f"{appendix_count}/{total_sections}"
            )

        if (
            total_sections >= 20
            and not has_model
        ):
            reasons.append(
                "未识别出任何模型建立/求解章节"
            )

        stat = {
            "paper_id": paper_id,
            "sections": total_sections,
            "chunks": len(cc),
            "section_types": dict(
                types
            ),
            "other_ratio": round(
                other_ratio,
                4
            ),
            "analysis_ratio": round(
                analysis_ratio,
                4
            ),
            "reference_ratio": round(
                reference_ratio,
                4
            ),
            "appendix_ratio": round(
                appendix_ratio,
                4
            ),
            "reasons": reasons
        }

        paper_stats.append(
            stat
        )

        if reasons:

            # 把异常类别的标题带出来，
            # 方便下一步直接看规则哪里漏了

            samples = {}

            for target_type in [
                "other",
                "analysis",
                "reference",
                "appendix"
            ]:

                titles = []

                for x in ss:

                    if (
                        x.get(
                            "section_type"
                        )
                        == target_type
                    ):

                        title = x.get(
                            "section_title",
                            ""
                        )

                        if (
                            title
                            and title
                            not in titles
                        ):
                            titles.append(
                                title
                            )

                if titles:

                    samples[
                        target_type
                    ] = titles[:40]

            suspicious_papers.append({
                **stat,
                "title_samples": samples
            })

    # ==================================================
    # 3. chunk 长度
    # ==================================================

    short_chunks = []
    long_atomic_chunks = []
    long_non_atomic_chunks = []

    for x in chunks:

        content = x.get(
            "content",
            ""
        )

        length = len(content)

        item = {
            "chunk_id":
                x.get("chunk_id"),
            "paper_id":
                x.get("paper_id"),
            "section_title":
                x.get(
                    "section_title"
                ),
            "section_type":
                x.get(
                    "section_type"
                ),
            "length":
                length
        }

        if length < 50:

            short_chunks.append(
                item
            )

        if length > 1000:

            if is_atomic_long_chunk(
                content
            ):

                long_atomic_chunks.append(
                    item
                )

            else:

                item[
                    "content_preview"
                ] = content[:500]

                long_non_atomic_chunks.append(
                    item
                )

    # ==================================================
    # 4. 表格/公式完整性
    # ==================================================

    broken_chunks = []

    for x in chunks:

        content = x.get(
            "content",
            ""
        )

        problems = []

        if (
            content.count("<table")
            !=
            content.count("</table>")
        ):
            problems.append(
                "table标签不平衡"
            )

        if (
            content.count("$$")
            % 2 != 0
        ):
            problems.append(
                "$$公式标记不平衡"
            )

        if problems:

            broken_chunks.append({
                "chunk_id":
                    x.get("chunk_id"),
                "paper_id":
                    x.get("paper_id"),
                "section_title":
                    x.get(
                        "section_title"
                    ),
                "problems":
                    problems
            })

    # ==================================================
    # 5. 重复 chunk
    # ==================================================

    content_map = defaultdict(list)

    for x in chunks:

        content = (
            x.get(
                "content",
                ""
            )
            .strip()
        )

        if not content:
            continue

        h = hashlib.md5(
            content.encode(
                "utf-8"
            )
        ).hexdigest()

        content_map[h].append({
            "chunk_id":
                x.get("chunk_id"),
            "paper_id":
                x.get("paper_id"),
            "section_title":
                x.get(
                    "section_title"
                )
        })

    duplicate_groups = [
        group
        for group
        in content_map.values()
        if len(group) > 1
    ]

    # ==================================================
    # 6. ID 唯一性
    # ==================================================

    section_ids = [
        x.get("section_id")
        for x in sections
    ]

    chunk_ids = [
        x.get("chunk_id")
        for x in chunks
    ]

    duplicate_section_ids = [
        x
        for x, cnt
        in Counter(
            section_ids
        ).items()
        if cnt > 1
    ]

    duplicate_chunk_ids = [
        x
        for x, cnt
        in Counter(
            chunk_ids
        ).items()
        if cnt > 1
    ]

    # ==================================================
    # 最终报告
    # ==================================================

    report = {

        "summary": {
            "paper_count":
                len(paper_sections),

            "section_count":
                len(sections),

            "chunk_count":
                len(chunks),

            "section_type_count":
                dict(
                    section_type_count
                ),

            "suspicious_paper_count":
                len(
                    suspicious_papers
                ),

            "short_chunk_count":
                len(short_chunks),

            "long_atomic_chunk_count":
                len(
                    long_atomic_chunks
                ),

            "long_non_atomic_chunk_count":
                len(
                    long_non_atomic_chunks
                ),

            "broken_chunk_count":
                len(
                    broken_chunks
                ),

            "duplicate_content_groups":
                len(
                    duplicate_groups
                ),

            "duplicate_section_id_count":
                len(
                    duplicate_section_ids
                ),

            "duplicate_chunk_id_count":
                len(
                    duplicate_chunk_ids
                )
        },

        "suspicious_papers":
            suspicious_papers,

        "long_non_atomic_chunks":
            long_non_atomic_chunks,

        "broken_chunks":
            broken_chunks,

        "duplicate_content_groups":
            duplicate_groups[:100],

        "duplicate_section_ids":
            duplicate_section_ids,

        "duplicate_chunk_ids":
            duplicate_chunk_ids,

        # 这里只保留一部分短 chunk，
        # 避免报告过大
        "short_chunk_samples":
            short_chunks[:100],

        "paper_stats":
            paper_stats
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ==================================================
    # 控制台摘要
    # ==================================================

    print()
    print("=" * 70)
    print("检查完成")
    print("=" * 70)

    print(
        "论文：",
        len(paper_sections)
    )

    print(
        "sections：",
        len(sections)
    )

    print(
        "chunks：",
        len(chunks)
    )

    print()

    print(
        "异常论文：",
        len(suspicious_papers)
    )

    for x in suspicious_papers:

        print(
            " -",
            x["paper_id"],
            " | ",
            "; ".join(
                x["reasons"]
            )
        )

    print()
    print(
        "短 chunk(<50)：",
        len(short_chunks)
    )

    print(
        "长 atomic chunk(>1000)：",
        len(long_atomic_chunks)
    )

    print(
        "长普通 chunk(>1000)：",
        len(long_non_atomic_chunks)
    )

    print(
        "表格/公式损坏：",
        len(broken_chunks)
    )

    print(
        "重复内容组：",
        len(duplicate_groups)
    )

    print(
        "重复 section_id：",
        len(
            duplicate_section_ids
        )
    )

    print(
        "重复 chunk_id：",
        len(
            duplicate_chunk_ids
        )
    )

    print()

    print(
        "报告保存到：",
        REPORT_PATH
    )


if __name__ == "__main__":
    main()