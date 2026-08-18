import json
from pathlib import Path

from split_paper import process_paper
from make_chunks import make_chunks


ROOT = Path("D:/math_model_kb")

PARSED_ROOT = ROOT / "parsed_all"
PROCESSED_ROOT = ROOT / "processed_all"

ALL_SECTIONS = PROCESSED_ROOT / "all_sections.jsonl"
ALL_CHUNKS = PROCESSED_ROOT / "all_chunks.jsonl"


def read_jsonl(path):
    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                result.append(
                    json.loads(line)
                )

    return result


def main():

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    papers = []

    # ==========================================
    # 找到所有论文目录
    # ==========================================

    for problem_dir in sorted(
        PARSED_ROOT.glob("CUMCM_*")
    ):

        for paper_dir in sorted(
            problem_dir.glob("paper_*")
        ):

            md_files = list(
                paper_dir.rglob("*.md")
            )

            if not md_files:
                print(
                    "✗ 没找到 Markdown：",
                    problem_dir.name,
                    paper_dir.name
                )
                continue

            if len(md_files) > 1:
                print(
                    "⚠ 找到多个 Markdown：",
                    problem_dir.name,
                    paper_dir.name
                )

            md_path = md_files[0]

            papers.append(
                (
                    problem_dir.name,
                    paper_dir.name,
                    md_path
                )
            )

    print(
        f"共找到 {len(papers)} 篇 Markdown"
    )

    print("=" * 70)

    all_sections = []
    all_chunks = []

    failed = []

    # ==========================================
    # 逐篇处理
    # ==========================================

    for index, (
        problem_id,
        paper_name,
        md_path
    ) in enumerate(
        papers,
        start=1
    ):

        # paper_001 -> 001
        paper_num = (
            paper_name
            .replace("paper_", "")
        )

        # CUMCM_2022_A_001
        paper_id = (
            f"{problem_id}_{paper_num}"
        )

        output_dir = (
            PROCESSED_ROOT
            / problem_id
            / paper_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        sections_path = (
            output_dir
            / "sections.jsonl"
        )

        chunks_path = (
            output_dir
            / "chunks.jsonl"
        )

        print()
        print("=" * 70)

        print(
            f"[{index}/{len(papers)}]"
        )

        print(
            "处理：",
            paper_id
        )

        print(
            "Markdown：",
            md_path
        )

        try:

            # ==============================
            # Markdown → sections
            # ==============================

            process_paper(
                input_path=str(md_path),
                output_path=str(
                    sections_path
                ),
                paper_id=paper_id,
                problem_id=problem_id
            )

            # ==============================
            # sections → chunks
            # ==============================

            make_chunks(
                input_path=str(
                    sections_path
                ),
                output_path=str(
                    chunks_path
                )
            )

            sections = read_jsonl(
                sections_path
            )

            chunks = read_jsonl(
                chunks_path
            )

            all_sections.extend(
                sections
            )

            all_chunks.extend(
                chunks
            )

            print(
                f"✓ {paper_id}"
                f" | sections={len(sections)}"
                f" | chunks={len(chunks)}"
            )

        except Exception as e:

            print(
                f"✗ {paper_id} 处理失败"
            )

            print(
                "原因：",
                repr(e)
            )

            failed.append({
                "paper_id": paper_id,
                "error": repr(e)
            })

    # ==========================================
    # 写总 sections
    # ==========================================

    with ALL_SECTIONS.open(
        "w",
        encoding="utf-8"
    ) as f:

        for item in all_sections:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ==========================================
    # 写总 chunks
    # ==========================================

    with ALL_CHUNKS.open(
        "w",
        encoding="utf-8"
    ) as f:

        for item in all_chunks:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ==========================================
    # 统计
    # ==========================================

    print()
    print("=" * 70)

    print("全部处理完成")

    print(
        "论文数量：",
        len(papers)
    )

    print(
        "成功：",
        len(papers) - len(failed)
    )

    print(
        "失败：",
        len(failed)
    )

    print(
        "总 sections：",
        len(all_sections)
    )

    print(
        "总 chunks：",
        len(all_chunks)
    )

    print(
        "总 sections 文件：",
        ALL_SECTIONS
    )

    print(
        "总 chunks 文件：",
        ALL_CHUNKS
    )

    if failed:

        failed_path = (
            PROCESSED_ROOT
            / "failed.json"
        )

        with failed_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                failed,
                f,
                ensure_ascii=False,
                indent=2
            )

        print()
        print(
            "失败记录：",
            failed_path
        )


if __name__ == "__main__":
    main()