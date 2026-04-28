"""
RAG-based Enterprise Document Chatbot Pipeline
================================================
가상의 중견기업 "모든전자(Modeun Electronics)" 사내 문서 5종(TXT / JSON / JSONL /
CSV / PDF)을 LangChain RAG 파이프라인에 흘려, 자연어 질문에 회사 내부 정보로
답하는 챗봇을 구현한 통합 스크립트입니다.

실행 모드:
    --mode stats        : 문서 통계 + 청크 통계 (LLM 호출 없음, 시각화 가능)
    --mode build        : 5개 컬렉션 생성 + 임베딩 + ChromaDB 저장
    --mode chat         : 5개 RAG Tool을 가진 Agent 데모
    --mode visualize    : 시각화 PNG 4종 저장
    --mode all          : 위 단계 전체 순차 실행

원본 노트북:
    chapter_04_rag.ipynb / codes/1_vector_store.py / codes/2_rag.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    JSONLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ────────────────────────────────────────────────────────────────────────────
# 0. 경로 / 폰트 / 상수
# ────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CHROMA_DIR = ROOT_DIR / "chroma_store"

RESULTS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHUNK_SIZE = 500


# ────────────────────────────────────────────────────────────────────────────
# 1. 데이터 소스 정의 — 5종 문서를 한 곳에서 관리
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class SourceSpec:
    key: str             # 컬렉션 이름 + Tool 이름 prefix
    label: str           # 사람이 읽을 한글 라벨
    file_name: str       # data/ 안의 파일 이름
    loader_kind: str     # text | json | jsonl | csv | pdf
    chunkable: bool      # JSON/JSONL은 통째로 임베딩 (구조 보존)
    description: str     # Agent에 전달할 Tool docstring 본문


SOURCES: list[SourceSpec] = [
    SourceSpec(
        key="company_profile",
        label="회사 개요",
        file_name="company_profile.txt",
        loader_kind="text",
        chunkable=True,
        description=(
            "모든전자의 설립 연혁·핵심 사업·해외 거점·디지털 서비스 등 회사 개요와 "
            "공식 프로필 정보를 검색합니다. 슬로건, 핵심 가치, 주요 연혁 같은 질문에 사용합니다."
        ),
    ),
    SourceSpec(
        key="tech_quality",
        label="기술·품질 역량",
        file_name="tech_quality.json",
        loader_kind="json",
        chunkable=False,
        description=(
            "융합 기술 연구소(R&D), 품질 관리 시스템(QMS), ISO/IATF 등 인증, FAE 지원 등 "
            "기술 및 품질 역량 정보를 검색합니다. 인증 일정·연구 분야·시뮬레이션 도구 같은 질문에 사용합니다."
        ),
    ),
    SourceSpec(
        key="products_services",
        label="제품·서비스",
        file_name="products_services.jsonl",
        loader_kind="jsonl",
        chunkable=False,
        description=(
            "ODM 제품, 유통 부품, 솔루션 서비스 각각의 ID·이름·카테고리·기술 사양·적용 분야 등 "
            "구체적인 제품 데이터를 검색합니다. 'GaN 모듈 효율은?' 같은 사양 질문에 사용합니다."
        ),
    ),
    SourceSpec(
        key="customer_satisfaction_metric",
        label="고객 만족도 지표",
        file_name="customer_satisfaction_metric.csv",
        loader_kind="csv",
        chunkable=True,
        description=(
            "주요 고객사별 납기 준수율(OTD), 품질 지수(PPM), 연간 거래액, 품질 감사 점수 등 "
            "고객 만족도와 계약 지표를 검색합니다. 산업·업종·실적 비교 질문에 사용합니다."
        ),
    ),
    SourceSpec(
        key="recruitment_process_guide",
        label="채용 프로세스",
        file_name="recruitment_process_guide.pdf",
        loader_kind="pdf",
        chunkable=True,
        description=(
            "신입/경력 채용 4단계 절차, 단계별 평가 기준, 직무별 추가 과제(코딩 테스트, 외국어 면접 등) "
            "정보를 검색합니다. 채용 일정·면접 가이드 질문에 사용합니다."
        ),
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# 2. Loader 매핑 — loader_kind에 따라 적절한 LangChain 로더 선택
# ────────────────────────────────────────────────────────────────────────────
def _build_text_loader(spec: SourceSpec) -> TextLoader:
    return TextLoader(str(DATA_DIR / spec.file_name), encoding="utf-8")


def _build_json_loader(spec: SourceSpec, json_lines: bool) -> JSONLoader:
    def _meta(_record: dict, metadata: dict) -> dict:
        metadata["source"] = str(DATA_DIR / spec.file_name)
        metadata["theme"] = spec.label
        return metadata

    return JSONLoader(
        file_path=str(DATA_DIR / spec.file_name),
        jq_schema=".",
        text_content=False,
        json_lines=json_lines,
        metadata_func=_meta,
    )


def _build_csv_loader(spec: SourceSpec) -> CSVLoader:
    return CSVLoader(
        file_path=str(DATA_DIR / spec.file_name),
        encoding="utf-8",
        metadata_columns=["customer_id", "industry"],
    )


def _build_pdf_loader(spec: SourceSpec) -> PyMuPDFLoader:
    return PyMuPDFLoader(str(DATA_DIR / spec.file_name))


def load_one(spec: SourceSpec) -> list[Document]:
    """5종 로더 분기를 하나의 함수로 통일."""
    if spec.loader_kind == "text":
        return _build_text_loader(spec).load()
    if spec.loader_kind == "json":
        return _build_json_loader(spec, json_lines=False).load()
    if spec.loader_kind == "jsonl":
        return _build_json_loader(spec, json_lines=True).load()
    if spec.loader_kind == "csv":
        return _build_csv_loader(spec).load()
    if spec.loader_kind == "pdf":
        return _build_pdf_loader(spec).load()
    raise ValueError(f"Unknown loader_kind: {spec.loader_kind}")


# ────────────────────────────────────────────────────────────────────────────
# 3. 청크 분할 — 구조화 데이터(JSON/JSONL)는 통째로 보존
# ────────────────────────────────────────────────────────────────────────────
def split_one(spec: SourceSpec, docs: list[Document]) -> list[Document]:
    if not spec.chunkable:
        return docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, add_start_index=True
    )
    return splitter.split_documents(docs)


# ────────────────────────────────────────────────────────────────────────────
# 4. 임베딩 + ChromaDB 저장
# ────────────────────────────────────────────────────────────────────────────
def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=os.environ["EMBEDDING_URL"],
        api_key=os.environ["API_KEY"],
        model="ignored-by-proxy",
        check_embedding_ctx_length=False,
    )


def build_vector_store(spec: SourceSpec, embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=spec.key,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def build_all_collections() -> dict[str, Chroma]:
    """5종 문서를 모두 로드 → 분할 → 임베딩 → 저장 후 컬렉션 dict 반환."""
    embeddings = build_embeddings()
    stores: dict[str, Chroma] = {}
    for spec in SOURCES:
        print(f"  [{spec.key}] loading & embedding ...")
        docs = load_one(spec)
        chunks = split_one(spec, docs)
        store = build_vector_store(spec, embeddings)
        store.add_documents(chunks)
        stores[spec.key] = store
        print(f"     → {len(docs)} docs → {len(chunks)} chunks 저장 완료")
    return stores


# ────────────────────────────────────────────────────────────────────────────
# 5. RAG Tool — 컬렉션 1개당 Tool 1개씩 자동 생성
# ────────────────────────────────────────────────────────────────────────────
def make_search_tool(spec: SourceSpec, store: Chroma) -> Callable:
    """spec의 docstring을 그대로 Tool 설명으로 노출하기 위한 클로저 빌더."""

    @tool(name_or_callable=f"search_{spec.key}", description=spec.description)
    def _search(query: str) -> str:
        retrieved = store.similarity_search(query, k=2)
        if not retrieved:
            return "관련 정보를 찾지 못했습니다."
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', spec.label)}]\n{doc.page_content}"
            for doc in retrieved
        )

    return _search


def register_tools(stores: dict[str, Chroma]) -> list[Callable]:
    return [make_search_tool(spec, stores[spec.key]) for spec in SOURCES]


# ────────────────────────────────────────────────────────────────────────────
# 6. Agent — 5개 Tool을 모두 들고 답변
# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
너는 전자부품 유통 및 솔루션 전문 기업 '모든전자(Modeun Electronics Co., Ltd.)'의 공식 챗봇이다.
항상 한국어로 답변한다.

다음 다섯 개 검색 툴을 보유한다.
- search_company_profile : 회사 개요·연혁·해외 거점·디지털 서비스 등 일반 정보
- search_tech_quality : R&D, 품질 관리, 인증, FAE 지원 등 기술·품질 역량
- search_products_services : ODM/유통/서비스 제품의 기술 사양·적용 분야
- search_customer_satisfaction_metric : 고객사 OTD/PPM/거래액/감사 점수
- search_recruitment_process_guide : 신입/경력 채용 절차 및 직무별 가이드

질문의 주제를 파악해 가장 적절한 툴 하나만 우선 호출한다.
답변은 검색된 문서 내용에 근거해 작성하며, 근거가 없으면 추측하지 않는다.
일반 인사·잡담은 툴을 사용하지 않고 답변한다.
"""


def build_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        model="ignored-by-proxy",
    )


def run_chat_demo(stores: dict[str, Chroma]) -> dict[str, Any]:
    llm = build_llm()
    tools = register_tools(stores)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    questions = [
        "모든전자의 슬로건과 핵심 가치를 알려줘.",
        "융합 기술 연구소의 주요 연구 분야는 뭐야?",
        "GaN 기반 초고밀도 전력 모듈의 효율은 몇 %야?",
        "납기 준수율이 가장 높은 산업군 고객사는 어디야?",
        "디지털 직무 채용에는 코딩 과제가 있어?",
    ]
    log: list[dict[str, str]] = []
    for q in questions:
        print(f"\n>>> Q: {q}")
        out = agent.invoke({"messages": [{"role": "user", "content": q}]})
        ans = out["messages"][-1].content
        print(f"<<< A: {ans[:240]}{'...' if len(ans) > 240 else ''}")
        log.append({"question": q, "answer": ans})
    return {"qa_log": log}


# ────────────────────────────────────────────────────────────────────────────
# 7. 통계 (LLM 호출 없이 가능)
# ────────────────────────────────────────────────────────────────────────────
def gather_stats() -> list[dict[str, Any]]:
    """각 소스별 (파일 크기, 원본 문서 수, 분할 후 청크 수, 평균 청크 길이) 계산."""
    stats: list[dict[str, Any]] = []
    for spec in SOURCES:
        path = DATA_DIR / spec.file_name
        size_kb = path.stat().st_size / 1024
        docs = load_one(spec)
        chunks = split_one(spec, docs)
        avg_len = (
            int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1))
        )
        stats.append(
            {
                "key": spec.key,
                "label": spec.label,
                "file_name": spec.file_name,
                "size_kb": round(size_kb, 1),
                "doc_count": len(docs),
                "chunk_count": len(chunks),
                "avg_chunk_len": avg_len,
                "loader": spec.loader_kind,
                "chunkable": spec.chunkable,
            }
        )
    return stats


# ────────────────────────────────────────────────────────────────────────────
# 8. 시각화
# ────────────────────────────────────────────────────────────────────────────
def visualize_pipeline_overview() -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.2, 2.3, 1.7, 1.4, "5종 문서\nTXT/JSON\nJSONL/CSV/PDF", "#FFE5B4"),
        (2.2, 2.3, 1.9, 1.4, "Document\nLoaders", "#FFD180"),
        (4.4, 2.3, 1.9, 1.4, "Recursive\nCharacterText\nSplitter", "#B3E5FC"),
        (6.6, 2.3, 1.9, 1.4, "OpenAI\nEmbeddings\n(별도 프록시)", "#C5CAE9"),
        (8.8, 2.3, 1.9, 1.4, "ChromaDB\n(5개 컬렉션)", "#C8E6C9"),
        (11.0, 4.0, 2.5, 0.9, "RAG Tool ×5\nsearch_*", "#FFCCBC"),
        (11.0, 0.9, 2.5, 0.9, "Agent\n(create_agent)", "#FFAB91"),
    ]
    for x, y, w, h, txt, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="black", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9.5, fontweight="bold")

    arrows_h = [(1.9, 3.0, 2.2, 3.0), (4.1, 3.0, 4.4, 3.0),
                (6.3, 3.0, 6.6, 3.0), (8.5, 3.0, 8.8, 3.0)]
    for x1, y1, x2, y2 in arrows_h:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.6))

    ax.annotate("", xy=(11.0, 4.45), xytext=(10.7, 3.4),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.6))
    ax.annotate("", xy=(12.25, 1.8), xytext=(12.25, 4.0),
                arrowprops=dict(arrowstyle="<->", color="#666", lw=1.4, linestyle="--"))

    ax.text(7, 5.5, "RAG 파이프라인 전체 흐름",
            ha="center", fontsize=14, fontweight="bold")
    legend = [
        mpatches.Patch(facecolor="#FFE5B4", edgecolor="black", label="원본 문서"),
        mpatches.Patch(facecolor="#FFD180", edgecolor="black", label="로드"),
        mpatches.Patch(facecolor="#B3E5FC", edgecolor="black", label="청크 분할"),
        mpatches.Patch(facecolor="#C5CAE9", edgecolor="black", label="임베딩"),
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="black", label="벡터 DB"),
        mpatches.Patch(facecolor="#FFCCBC", edgecolor="black", label="Tool / Agent"),
    ]
    ax.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=9)

    out = RESULTS_DIR / "fig_01_pipeline_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_document_stats(stats: list[dict[str, Any]]) -> Path:
    labels = [s["label"] for s in stats]
    file_kb = [s["size_kb"] for s in stats]
    docs = [s["doc_count"] for s in stats]
    chunks = [s["chunk_count"] for s in stats]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].barh(labels, file_kb, color="#FFB74D", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(file_kb):
        axes[0].text(v + max(file_kb) * 0.02, i, f"{v:,.1f} KB",
                     va="center", fontsize=9)
    axes[0].set_xlabel("파일 크기 (KB)")
    axes[0].set_title("원본 파일 크기", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", linestyle=":", alpha=0.4)

    axes[1].barh(labels, docs, color="#4FC3F7", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(docs):
        axes[1].text(v + max(docs) * 0.02, i, str(v), va="center", fontsize=9)
    axes[1].set_xlabel("Document 객체 수 (Loader 출력)")
    axes[1].set_title("로드 직후 Document 수", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", linestyle=":", alpha=0.4)

    axes[2].barh(labels, chunks, color="#81C784", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(chunks):
        axes[2].text(v + max(chunks) * 0.02, i, str(v), va="center", fontsize=9)
    axes[2].set_xlabel("최종 청크 수 (벡터 DB에 저장된 단위)")
    axes[2].set_title("Splitter 통과 후 청크 수", fontweight="bold")
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("문서 5종 — Loader · Splitter 통계", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = RESULTS_DIR / "fig_02_document_stats.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def visualize_chunk_length_distribution() -> Path:
    """청크 가능한 소스의 청크 길이 히스토그램."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"company_profile": "#1976D2", "customer_satisfaction_metric": "#388E3C",
              "recruitment_process_guide": "#D32F2F"}
    for spec in SOURCES:
        if not spec.chunkable:
            continue
        chunks = split_one(spec, load_one(spec))
        lengths = [len(c.page_content) for c in chunks]
        ax.hist(lengths, bins=12, alpha=0.6, label=spec.label,
                color=colors.get(spec.key, "#888"), edgecolor="black", linewidth=0.4)

    ax.axvline(CHUNK_SIZE, color="red", linestyle="--", linewidth=1.2,
               label=f"chunk_size 상한 = {CHUNK_SIZE}")
    ax.set_xlabel("청크 길이 (문자 수)")
    ax.set_ylabel("청크 개수")
    ax.set_title("청크 가능한 3개 소스의 청크 길이 분포",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    out = RESULTS_DIR / "fig_03_chunk_length_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def visualize_tool_routing_map() -> Path:
    """질문 유형 → 어떤 Tool이 호출되는지 라우팅 맵."""
    routing = [
        ("회사 슬로건이 뭐야?", "search_company_profile"),
        ("ISO 인증 일정은?", "search_tech_quality"),
        ("GaN 모듈 효율은?", "search_products_services"),
        ("PPM 가장 낮은 고객사?", "search_customer_satisfaction_metric"),
        ("코딩 테스트 있어?", "search_recruitment_process_guide"),
        ("해외 거점 알려줘", "search_company_profile"),
        ("FAE 전문 분야?", "search_tech_quality"),
        ("CAN 프로토콜 제품?", "search_products_services"),
    ]
    questions = [r[0] for r in routing]
    tools_set = list(dict.fromkeys(r[1] for r in routing))
    matrix = np.zeros((len(questions), len(tools_set)))
    for i, (_, t) in enumerate(routing):
        matrix[i, tools_set.index(t)] = 1

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.imshow(matrix, cmap="Greens", aspect="auto", vmin=0, vmax=1.5)
    ax.set_xticks(range(len(tools_set)))
    ax.set_xticklabels([t.replace("search_", "") for t in tools_set],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=9)
    for i in range(len(questions)):
        for j in range(len(tools_set)):
            if matrix[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
    ax.set_title("질문 유형 → 호출되는 RAG Tool",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    out = RESULTS_DIR / "fig_04_tool_routing_map.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모든전자 RAG 챗봇 파이프라인")
    p.add_argument(
        "--mode",
        choices=["stats", "build", "chat", "visualize", "all"],
        default="all",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = {}

    if args.mode in {"stats", "visualize", "all"}:
        print("\n[ 1. 문서 / 청크 통계 ]")
        stats = gather_stats()
        for s in stats:
            print(f"  - {s['label']:<14}({s['loader']:>5}): "
                  f"{s['size_kb']:>6.1f} KB | docs={s['doc_count']:>3} | "
                  f"chunks={s['chunk_count']:>3} | avg_len={s['avg_chunk_len']:>4}")
        summary["stats"] = stats

    stores: dict[str, Chroma] | None = None
    if args.mode in {"build", "chat", "all"}:
        print("\n[ 2. 임베딩 + ChromaDB 구축 ]")
        load_dotenv()
        stores = build_all_collections()

    if args.mode in {"chat", "all"} and stores is not None:
        print("\n[ 3. RAG Agent 챗봇 데모 ]")
        summary.update(run_chat_demo(stores))

    if args.mode in {"visualize", "all"}:
        print("\n[ 4. 시각화 저장 ]")
        stats_for_plot = summary.get("stats") or gather_stats()
        paths = [
            visualize_pipeline_overview(),
            visualize_document_stats(stats_for_plot),
            visualize_chunk_length_distribution(),
            visualize_tool_routing_map(),
        ]
        for p in paths:
            print(f"  - saved: {p.relative_to(ROOT_DIR)}")

    if summary:
        out_json = RESULTS_DIR / "rag_run_log.json"
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n실행 로그 저장: {out_json.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
