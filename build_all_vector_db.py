import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


ROOT = Path("D:/math_model_kb")

CHUNKS_PATH = (
    ROOT
    / "processed_all"
    / "all_chunks.jsonl"
)

DB_PATH = (
    ROOT
    / "database"
    / "chroma"
)

MODEL_NAME = str(
    ROOT
    / "models"
    / "bge-base-zh-v1.5"
)

COLLECTION_NAME = "math_model_papers_all_v1"

BATCH_SIZE = 128


def read_jsonl(path):
    result = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                result.append(
                    json.loads(line)
                )

    return result


def parse_problem_id(problem_id):
    # CUMCM_2024_A
    parts = problem_id.split("_")

    year = None
    problem_code = None

    if len(parts) >= 3:
        try:
            year = int(parts[1])
        except ValueError:
            pass

        problem_code = parts[2]

    return year, problem_code


def build_search_text(item):
    parts = []

    parts.append(
        f"数学建模优秀论文：{item['paper_id']}"
    )

    parts.append(
        f"赛题：{item['problem_id']}"
    )

    if item.get("question_number") is not None:
        parts.append(
            f"问题{item['question_number']}"
        )

    parts.append(
        f"论文模块：{item.get('main_section', '')}"
    )

    parts.append(
        f"章节类型：{item.get('section_type', '')}"
    )

    parts.append(
        f"章节标题：{item.get('section_title', '')}"
    )

    parts.append("")
    parts.append(
        item.get("content", "")
    )

    return "\n".join(parts)


def build_metadata(item):
    problem_id = item["problem_id"]

    year, problem_code = (
        parse_problem_id(problem_id)
    )

    metadata = {
        "source_type": "paper",
        "competition": "CUMCM",
        "paper_id": item["paper_id"],
        "problem_id": problem_id,
        "section_id": item["section_id"],
        "section_title": item.get(
            "section_title",
            "",
        ),
        "section_type": item.get(
            "section_type",
            "other",
        ),
        "main_section": item.get(
            "main_section",
            "other",
        ),
        "chunk_index": int(
            item.get(
                "chunk_index",
                1,
            )
        ),
        "char_count": int(
            item.get(
                "char_count",
                len(item.get("content", "")),
            )
        ),
    }

    if year is not None:
        metadata["year"] = year

    if problem_code:
        metadata["problem_code"] = (
            problem_code
        )

    if item.get(
        "question_number"
    ) is not None:
        metadata["question_number"] = int(
            item["question_number"]
        )

    return metadata


def main():

    print("正在读取全部 chunks...")

    chunks = read_jsonl(
        CHUNKS_PATH
    )

    print(
        f"读取完成：{len(chunks)} 个 chunks"
    )

    # 最后一道保险：
    # 即使 make_chunks.py 中已排除，
    # 这里仍不让 reference / appendix 进入向量库。
    before = len(chunks)

    chunks = [
        x
        for x in chunks
        if x.get("section_type")
        not in {
            "reference",
            "appendix",
        }
    ]

    removed = before - len(chunks)

    print(
        f"过滤 reference/appendix："
        f"{removed} 个"
    )

    print(
        f"实际准备入库：{len(chunks)} 个"
    )

    print("正在加载 embedding 模型...")

    embedding_function = (
        SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
            device="cpu",
            normalize_embeddings=True,
        )
    )

    DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = (
        client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=(
                embedding_function
            ),
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
            metadata={
                "description":
                    "36篇CUMCM优秀数学建模论文知识库"
            },
        )
    )

    total = len(chunks)

    print("开始分批写入 Chroma...")

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        batch = chunks[
            start:
            start + BATCH_SIZE
        ]

        ids = [
            x["chunk_id"]
            for x in batch
        ]

        documents = [
            build_search_text(x)
            for x in batch
        ]

        metadatas = [
            build_metadata(x)
            for x in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        end = min(
            start + len(batch),
            total,
        )

        print(
            f"[{end}/{total}] 已写入"
        )

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        "Collection：",
        COLLECTION_NAME,
    )

    print(
        "当前记录数量：",
        collection.count(),
    )

    print(
        "数据库位置：",
        DB_PATH,
    )


if __name__ == "__main__":
    main()
