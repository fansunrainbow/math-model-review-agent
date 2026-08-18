import json
import re
from pathlib import Path


MAX_CHARS = 900
MIN_CHARS = 120
OVERLAP_CHARS = 80


def read_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                data.append(json.loads(line))

    return data


def clean_text(text):
    text = text.replace("学生在线", "")
    text = text.replace("数学建模老哥", "")

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# 安全拆 HTML 表格
# =========================================================

def split_table(table_html, max_chars=MAX_CHARS):

    if len(table_html) <= max_chars:
        return [table_html]

    rows = re.findall(
        r"<tr>.*?</tr>",
        table_html,
        flags=re.S
    )

    # 找不到行就保持原样
    if not rows:
        return [table_html]

    # 默认第一行为表头
    header = rows[0]

    chunks = []
    current_rows = [header]

    for row in rows[1:]:

        candidate = (
            "<table>"
            + "".join(current_rows + [row])
            + "</table>"
        )

        if (
            len(candidate) > max_chars
            and len(current_rows) > 1
        ):

            chunks.append(
                "<table>"
                + "".join(current_rows)
                + "</table>"
            )

            # 新表重复表头
            current_rows = [
                header,
                row
            ]

        else:
            current_rows.append(row)

    if current_rows:

        chunks.append(
            "<table>"
            + "".join(current_rows)
            + "</table>"
        )

    return chunks


# =========================================================
# 普通文本过长时按句子拆
# =========================================================

def split_plain_text(
    text,
    max_chars=MAX_CHARS
):

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[。！？；!?;])",
        text
    )

    result = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(current) + len(sentence) <= max_chars:

            current += sentence

        else:

            if current:
                result.append(current)
                current = ""

            # 单句话本身太长
            if len(sentence) > max_chars:

                start = 0

                while start < len(sentence):

                    end = min(
                        start + max_chars,
                        len(sentence)
                    )

                    result.append(
                        sentence[start:end]
                    )

                    if end >= len(sentence):
                        break

                    start = (
                        end - OVERLAP_CHARS
                    )

            else:
                current = sentence

    if current:
        result.append(current)

    return result


# =========================================================
# 把 section 转成原子块
#
# 表格和公式不会被拆
# =========================================================

PROTECTED_PATTERN = re.compile(
    r"(<table>.*?</table>|\$\$.*?\$\$)",
    flags=re.S
)


def make_units(text):

    text = clean_text(text)

    parts = PROTECTED_PATTERN.split(text)

    units = []

    for part in parts:

        if not part:
            continue

        part = part.strip()

        if not part:
            continue

        # HTML 表格
        if part.startswith("<table>"):

            units.extend(
                split_table(part)
            )

        # 数学公式
        elif (
            part.startswith("$$")
            and part.endswith("$$")
        ):

            # 保证整条公式不被切断
            units.append(part)

        else:

            # 普通文字先按自然段
            paragraphs = re.split(
                r"\n\s*\n",
                part
            )

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                units.extend(
                    split_plain_text(
                        paragraph
                    )
                )

    return units


# =========================================================
# 把原子块组合成 chunk
# =========================================================

def pack_units(
    units,
    max_chars=MAX_CHARS
):

    chunks = []

    current = []

    for unit in units:

        # 特殊块本身就超过限制，例如特别大的公式
        # 宁可保持完整，也不硬切
        if len(unit) > max_chars:

            if current:

                chunks.append(
                    "\n\n".join(current)
                )

                current = []

            chunks.append(unit)

            continue

        if current:

            candidate = (
                "\n\n".join(
                    current + [unit]
                )
            )

        else:
            candidate = unit

        if len(candidate) <= max_chars:

            current.append(unit)

        else:

            if current:

                chunks.append(
                    "\n\n".join(current)
                )

            current = [unit]

    if current:

        chunks.append(
            "\n\n".join(current)
        )

    # ---------------------------------
    # 尝试合并特别短的 chunk
    # 但绝不跨 section
    # ---------------------------------

    changed = True

    while changed and len(chunks) > 1:

        changed = False

        for i in range(len(chunks)):

            if len(chunks[i]) >= MIN_CHARS:
                continue

            # 优先和后面合并
            if i + 1 < len(chunks):

                candidate = (
                    chunks[i]
                    + "\n\n"
                    + chunks[i + 1]
                )

                if len(candidate) <= max_chars:

                    chunks[i + 1] = candidate
                    chunks.pop(i)

                    changed = True
                    break

            # 否则和前面合并
            if i > 0:

                candidate = (
                    chunks[i - 1]
                    + "\n\n"
                    + chunks[i]
                )

                if len(candidate) <= max_chars:

                    chunks[i - 1] = candidate
                    chunks.pop(i)

                    changed = True
                    break

    return chunks


# =========================================================
# 主程序
# =========================================================

def make_chunks(
    input_path,
    output_path
):

    sections = read_jsonl(input_path)

    result = []

    for section in sections:
        if section.get("section_type") in {
            "reference",
            "appendix"
        }:
            continue
        # 正文知识库暂时不要附录和参考文献
        if section["section_type"] in {
            "reference",
            "appendix"
        }:
            continue

        units = make_units(
            section["content"]
        )

        pieces = pack_units(units)

        for index, piece in enumerate(
            pieces,
            start=1
        ):

            item = {

                "chunk_id":
                    f"{section['section_id']}"
                    f"_C{index:03d}",

                # 新增
                "section_id":
                    section["section_id"],

                "paper_id":
                    section["paper_id"],

                "problem_id":
                    section["problem_id"],

                "main_section":
                    section["main_section"],

                "section_type":
                    section["section_type"],

                "section_title":
                    section["section_title"],

                "question_number":
                    section["question_number"],

                "chunk_index":
                    index,

                "char_count":
                    len(piece),

                "content":
                    piece
            }

            result.append(item)

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output.open(
        "w",
        encoding="utf-8"
    ) as f:

        for item in result:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    lengths = [
        len(x["content"])
        for x in result
    ]

    print(
        f"处理完成，共生成 {len(result)} 个 chunks"
    )

    if lengths:

        print(
            f"平均长度："
            f"{sum(lengths) / len(lengths):.1f}"
        )

        print(
            f"最短长度：{min(lengths)}"
        )

        print(
            f"最长长度：{max(lengths)}"
        )


if __name__ == "__main__":

    make_chunks(

        input_path=(
            "C:/Users/32077/Desktop/"
            "math_model_kb/processed/"
            "CUMCM_2024_A_001_sections_v2.jsonl"
        ),

        output_path=(
            "C:/Users/32077/Desktop/"
            "math_model_kb/processed/"
            "CUMCM_2024_A_001_chunks_v2.jsonl"
        )
    )