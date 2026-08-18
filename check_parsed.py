from pathlib import Path
import json

ROOT = Path("D:/math_model_kb")
PARSED_ROOT = ROOT / "parsed_all"

results = []

for problem_dir in sorted(PARSED_ROOT.glob("CUMCM_*")):

    for paper_dir in sorted(problem_dir.glob("paper_*")):

        md_files = list(paper_dir.rglob("*.md"))

        item = {
            "problem_id": problem_dir.name,
            "paper_id": paper_dir.name,
            "markdown_count": len(md_files),
            "markdown_files": []
        }

        for md in md_files:
            try:
                size = md.stat().st_size
                text = md.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                item["markdown_files"].append({
                    "path": str(
                        md.relative_to(ROOT)
                    ),
                    "size_bytes": size,
                    "chars": len(text)
                })

            except Exception as e:
                item["markdown_files"].append({
                    "path": str(md),
                    "error": str(e)
                })

        results.append(item)


output = ROOT / "parsed_check.json"

with output.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


total = len(results)

success = sum(
    1
    for x in results
    if x["markdown_count"] > 0
)

failed = [
    x
    for x in results
    if x["markdown_count"] == 0
]

print("=" * 60)

print("检测到论文目录：", total)
print("成功生成 Markdown：", success)
print("没有 Markdown：", len(failed))

if failed:

    print("\n以下论文可能解析失败：")

    for x in failed:
        print(
            x["problem_id"],
            x["paper_id"]
        )

print()
print(
    "检查结果已保存：",
    output
)