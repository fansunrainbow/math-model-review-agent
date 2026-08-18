from pathlib import Path
import subprocess


ROOT = Path("D:/math_model_kb")

PARSED_ROOT = ROOT / "parsed_all"


def main():

    pdfs = []

    for pdf in ROOT.glob(
        "CUMCM_*/papers/paper_*.pdf"
    ):
        pdfs.append(pdf)

    pdfs.sort()

    print(
        f"共找到 {len(pdfs)} 篇论文"
    )

    for index, pdf in enumerate(
        pdfs,
        start=1
    ):

        problem_id = (
            pdf.parent.parent.name
        )

        paper_name = pdf.stem

        output_dir = (
            PARSED_ROOT
            / problem_id
            / paper_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("=" * 70)

        print(
            f"[{index}/{len(pdfs)}]"
        )

        print(
            "正在处理：",
            problem_id,
            paper_name
        )

        print("=" * 70)

        command = [
            "mineru",
            "-p",
            str(pdf),
            "-o",
            str(output_dir),
            "-b",
            "pipeline"
        ]

        result = subprocess.run(
            command
        )

        if result.returncode == 0:

            print(
                f"✓ 完成："
                f"{problem_id}/{paper_name}"
            )

        else:

            print(
                f"✗ 失败："
                f"{problem_id}/{paper_name}"
            )


if __name__ == "__main__":
    main()