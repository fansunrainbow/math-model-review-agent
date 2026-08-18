import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("D:/math_model_kb")
MAIN_SCRIPT = ROOT / "review_agent_main.py"
PARSE_ROOT = ROOT / "review_pdf_cache"


def find_mineru():
    current_python = Path(sys.executable)

    candidates = [
        current_python.with_name("mineru.exe"),
        current_python.with_name("mineru"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    found = shutil.which("mineru")
    if found:
        return found

    raise FileNotFoundError(
        "没有找到 mineru。请确认当前虚拟环境中已安装 MinerU。"
    )


def choose_markdown(output_dir: Path, pdf_stem: str):
    md_files = list(output_dir.rglob("*.md"))

    if not md_files:
        raise FileNotFoundError(
            f"在 {output_dir} 下没有找到 Markdown。"
        )

    same_name = [
        p for p in md_files
        if p.stem.lower() == pdf_stem.lower()
    ]

    if same_name:
        return max(
            same_name,
            key=lambda p: p.stat().st_size,
        )

    return max(
        md_files,
        key=lambda p: p.stat().st_size,
    )


def convert_pdf_to_markdown(pdf_path: Path):
    mineru = find_mineru()

    PARSE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir = PARSE_ROOT / pdf_path.stem
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        existing_md = choose_markdown(
            output_dir,
            pdf_path.stem,
        )

        print("检测到已有 MinerU 解析结果，直接复用：")
        print(existing_md)

        return existing_md

    except FileNotFoundError:
        pass

    print()
    print("=" * 72)
    print("正在使用 MinerU 解析 PDF...")
    print("=" * 72)
    print("PDF：", pdf_path)
    print("输出目录：", output_dir)
    print("MinerU：", mineru)

    env = os.environ.copy()
    env.setdefault(
        "MINERU_MODEL_SOURCE",
        "modelscope",
    )

    result = subprocess.run(
        [
            mineru,
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-b",
            "pipeline",
        ],
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "MinerU 解析失败，退出码："
            f"{result.returncode}"
        )

    md_path = choose_markdown(
        output_dir,
        pdf_path.stem,
    )

    print()
    print("MinerU 解析完成。")
    print("自动找到 Markdown：")
    print(md_path)

    return md_path


def run_main(md_path: Path, source_pdf: Path | None = None):
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"找不到主程序：{MAIN_SCRIPT}"
        )

    print()
    print("=" * 72)
    print("开始调用 Whole-Paper Review Agent Main")
    print("=" * 72)
    print("Markdown：", md_path)
    if source_pdf is not None:
        print("原始 PDF：", source_pdf)
    print()

    env = os.environ.copy()

    if source_pdf is not None:
        env["MATH_MODEL_PARSED_FROM_PDF"] = "1"
        env["MATH_MODEL_SOURCE_PDF"] = str(
            source_pdf.resolve()
        )
    else:
        # 防止 IDE/父进程残留变量使直接 MD/TXT 输入误启用 Guard。
        env.pop("MATH_MODEL_PARSED_FROM_PDF", None)
        env.pop("MATH_MODEL_SOURCE_PDF", None)

    result = subprocess.run(
        [
            sys.executable,
            str(MAIN_SCRIPT),
            str(md_path),
        ],
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "主评审程序失败，退出码："
            f"{result.returncode}"
        )


def main():
    if len(sys.argv) < 2:
        print(
            "用法：\n"
            "python review_pdf_main.py \"你的论文.pdf\""
        )
        sys.exit(1)

    source = Path(sys.argv[1]).resolve()

    if not source.exists():
        raise FileNotFoundError(
            f"文件不存在：{source}"
        )

    suffix = source.suffix.lower()

    if suffix == ".pdf":
        md_path = convert_pdf_to_markdown(
            source
        )
        run_main(
            md_path,
            source_pdf=source,
        )
        return

    if suffix in {
        ".md",
        ".markdown",
        ".txt",
    }:
        run_main(source)
        return

    raise ValueError(
        "目前支持：PDF / MD / Markdown / TXT"
    )


if __name__ == "__main__":
    main()
