# 📚 LangChain RAG Enterprise Document Chatbot
### **5종 사내 문서(TXT/JSON/JSONL/CSV/PDF) → 청크 분할 → 임베딩 → ChromaDB → 자연어 검색** — 회사 내부 정보로 답하는 RAG 챗봇 파이프라인

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logo=databricks&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-009688?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 프로젝트 요약 (Project Overview)

LLM(언어 모델)은 학습이 끝난 시점까지의 정보만 알고, 그 이후의 일이나 특정 회사의 내부 자료처럼 학습에 포함되지 않은 내용은 알지 못합니다. "우리 회사 슬로건이 뭐야?", "GaN 모듈 효율은 몇 %야?" 같은 질문은 모델 혼자서는 절대 답할 수 없습니다. 이런 한계를 해결하는 가장 대표적인 방법이 **RAG(Retrieval-Augmented Generation, 검색 기반 답변 생성)** 입니다. 미리 회사 내부 문서를 잘게 잘라 벡터로 바꿔 저장해 두고, 질문이 들어왔을 때 가장 비슷한 문서 조각을 찾아 LLM에게 함께 넘겨 답을 만들게 하는 구조입니다.

이 프로젝트는 가상의 회사 "모든전자(Modeun Electronics)"의 사내 문서 5종(회사 개요 TXT, 기술/품질 JSON, 제품 JSONL, 고객 만족도 CSV, 채용 가이드 PDF)을 모두 RAG 파이프라인에 흘려 보고, 자연어 질문이 들어왔을 때 챗봇이 알아서 적절한 문서 컬렉션을 검색해 답하는 구조를 직접 만들어 본 기록입니다. 단순히 "되더라" 수준에 그치지 않고, **JSON 같은 구조화 데이터는 왜 청크로 자르면 안 되는지 / 컬렉션을 5개로 나누면 어떤 점이 좋아지는지 / 임베딩 서버는 왜 LLM 서버와 다른 곳에 둘 수 있는지** 같은 설계 질문을 하나씩 손으로 풀어 본 것이 핵심 목표였습니다.

---

## 🎯 핵심 목표 (Motivation)

| 핵심 역량 &emsp;&emsp;&emsp; | 상세 목표 |
| :--- | :--- |
| **다양한 형식 문서 다루기** | 5가지 형식(TXT/JSON/JSONL/CSV/PDF)을 각각 알맞은 LangChain Loader로 읽어, 모두 같은 `Document` 객체로 통일 |
| **청크 분할 전략** | 일반 텍스트는 `RecursiveCharacterTextSplitter(chunk_size=500)`로 잘게 자르고, JSON/JSONL 같은 구조화 데이터는 통째로 보존하는 두 가지 전략을 분기 처리 |
| **벡터 DB 컬렉션 분리** | 컬렉션을 1개로 합치지 않고 **소스별 5개**로 나눠, 각 컬렉션에 1:1로 매칭되는 검색 함수를 자동 생성 |
| **임베딩/LLM 서버 분리** | 임베딩 전용 프록시(`EMBEDDING_URL`)와 LLM 프록시(`BASE_URL`)를 별도 환경 변수로 두어, 운영 시 두 서버를 따로 관리할 수 있게 설계 |
| **자연어 라우팅** | 시스템 프롬프트에 5개 함수의 역할을 자세히 적어, 질문 주제에 맞는 함수 하나를 LLM이 스스로 고르도록 유도 |

---

## 📂 프로젝트 구조 (Project Structure)

```text
23. langchain-rag-enterprise-chatbot/
├─ data/                                     # 모든전자 사내 문서 5종
│  ├─ company_profile.txt                    # 회사 개요·연혁·해외 거점 (TXT)
│  ├─ tech_quality.json                      # R&D / 품질 관리 / 인증 (JSON, 단일 객체)
│  ├─ products_services.jsonl                # 제품/서비스 6종 (JSONL, 한 줄당 한 제품)
│  ├─ customer_satisfaction_metric.csv       # 고객사 6곳 OTD/PPM/거래액 (CSV)
│  └─ recruitment_process_guide.pdf          # 채용 4단계 절차 가이드 (PDF, 멀티 페이지)
├─ results/
│  ├─ fig_01_pipeline_overview.png           # RAG 파이프라인 전체 흐름
│  ├─ fig_02_document_stats.png              # 5종 문서 — 파일 크기 / Document 수 / 청크 수
│  ├─ fig_03_chunk_length_distribution.png   # 청크 길이 분포 히스토그램
│  ├─ fig_04_tool_routing_map.png            # 질문 유형 → 호출 Tool 라우팅 맵
│  └─ rag_run_log.json                       # 실행 통계 + 챗봇 Q&A 로그
├─ chroma_store/                             # ChromaDB 영구 저장 폴더 (실행 시 자동 생성, gitignore)
├─ src/
│  └─ rag_pipeline.py                        # 통합 실행 스크립트 (mode 인자로 단계별 실행)
├─ .gitignore
├─ README.md
└─ requirements.txt
```

> **Note**: 프로젝트 루트에 `.env` 파일을 만들어 `BASE_URL`, `EMBEDDING_URL`, `API_KEY` 세 개를 채워 넣어야 임베딩과 챗봇 모드가 동작합니다 (`--mode stats` / `--mode visualize` 만 단독 실행 시에는 불필요).

---

## 🏗️ Architecture & 핵심 구현 (Architecture & Core Implementation)

### 1. RAG 파이프라인 전체 흐름

| RAG 파이프라인 (도식) |
| :---: |
| ![pipeline](results/fig_01_pipeline_overview.png) |

> 5종 원본 문서가 로더를 거쳐 `Document` 객체로 통일되고, 일반 텍스트는 청크로 잘게 잘립니다. 그 후 임베딩 서버에 보내 벡터로 바꾼 뒤 ChromaDB의 5개 컬렉션에 나뉘어 저장됩니다. 질문이 들어오면 Agent가 5개 검색 함수 중 하나를 골라 가장 비슷한 문서 조각을 받아 답변을 만듭니다.

### 2. 5종 문서별 처리 방법

| 문서 종류 &emsp; | 파일 &emsp;&emsp;&emsp;&emsp;&emsp; | 사용 Loader &emsp;&emsp; | 청크 분할 | 이유 |
| :---: | :--- | :--- | :---: | :--- |
| 회사 개요 | `company_profile.txt` | `TextLoader` | ✓ | 줄글이라 길이가 길어 청크 단위로 나눠야 검색 정확도가 올라감 |
| 기술·품질 | `tech_quality.json` | `JSONLoader`<br/>(`json_lines=False`) | ✗ | 한 덩어리 JSON 객체 — 자르면 의미가 깨지므로 통째로 임베딩 |
| 제품·서비스 | `products_services.jsonl` | `JSONLoader`<br/>(`json_lines=True`) | ✗ | 한 줄당 한 제품 — 이미 잘게 나뉘어 있어 추가 분할 불필요 |
| 고객 만족도 | `customer_satisfaction_metric.csv` | `CSVLoader` | ✓ | CSV 한 줄을 한 Document로 받지만, 짧은 줄 여러 개를 묶어 청크로 만듦 |
| 채용 가이드 | `recruitment_process_guide.pdf` | `PyMuPDFLoader` | ✓ | 페이지별로 길이가 다른 PDF — 청크로 잘라 검색 단위 통일 |

### 3. 핵심 설계 포인트

| 설계 포인트 &emsp;&emsp;&emsp;&emsp; | 적용 방법과 효과 |
| :--- | :--- |
| **컬렉션 5개 분리** | 모든 문서를 한 컬렉션에 넣지 않고 소스별로 5개 컬렉션을 만듦. 컬렉션마다 1:1로 매칭되는 검색 함수를 만들어, Agent가 "이 질문은 채용 관련이야"를 알면 채용 컬렉션만 검색하도록 유도 — 검색 잡음을 크게 줄임 |
| **구조화 데이터는 통째로** | JSON/JSONL은 `chunkable=False` 플래그로 분리해 청크 분할을 건너뜀. 데이터 안의 키-값 관계가 깨지지 않아 "GaN 모듈 효율" 같은 정밀한 사양 질문에도 정확히 응답 |
| **임베딩 서버 분리** | LLM 호출은 `BASE_URL`, 임베딩 호출은 `EMBEDDING_URL`로 따로 둠. 운영 환경에서 임베딩 서버만 별도로 확장하거나 다른 모델로 교체할 수 있는 여지를 남김 |
| **`SourceSpec` 한 곳 관리** | 5종 문서의 파일명·로더 종류·청크 여부·함수 설명을 하나의 데이터 클래스(`SourceSpec`)에 모아 둠. 새 문서를 추가하려면 항목 한 줄만 늘리면 끝나는 구조 |
| **Tool 자동 생성** | 컬렉션마다 검색 함수를 직접 손으로 5개 짜지 않고, `make_search_tool()` 클로저로 5개를 자동 생성. `SourceSpec`의 설명문이 그대로 함수 docstring이 되어 LLM에 노출됨 |

### 4. 실행 방법

| 명령어 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp; | 동작 &emsp;&emsp;&emsp;&emsp; | LLM/임베딩 호출 |
| :--- | :--- | :---: |
| `python src/rag_pipeline.py --mode stats` | 5종 문서의 파일 크기·문서 수·청크 수 통계만 출력 | ✗ |
| `python src/rag_pipeline.py --mode build` | 5개 컬렉션 생성 + 임베딩 + ChromaDB 영구 저장 | ✓ |
| `python src/rag_pipeline.py --mode chat` | 5개 검색 함수를 가진 챗봇으로 5개 질문 시연 | ✓ |
| `python src/rag_pipeline.py --mode visualize` | 시각화 PNG 4장 만들기 (API 호출 불필요) | ✗ |
| `python src/rag_pipeline.py --mode all` | 위 단계 전부 한 번에 실행 | ✓ |

---

## 📊 시각화 결과 (Results)

### 1. 5종 문서 — Loader · Splitter 통계

![document stats](results/fig_02_document_stats.png)

| 항목 &emsp;&emsp;&emsp;&emsp; | 내용 |
| :--- | :--- |
| **구성** | 왼쪽: 원본 파일 크기 (KB), 가운데: 로더 통과 직후의 Document 객체 수, 오른쪽: Splitter 통과 후 최종 청크 수 |
| **확인 포인트** | 채용 PDF는 134.8 KB로 가장 무거운데도 청크 수는 3개로 적음 (이미지·표가 많아 텍스트 비중이 작기 때문) / CSV는 0.5 KB이지만 한 줄당 한 Document로 6개가 만들어짐 / JSON은 1개, JSONL은 6개로 줄 단위가 그대로 Document 단위가 됨 |
| **의미** | "파일 크기 ≠ 검색 단위"라는 점을 시각적으로 확인. 검색 정확도를 결정하는 건 결국 가장 오른쪽 그래프의 **최종 청크 수**임 |

### 2. 청크 길이 분포

![chunk length distribution](results/fig_03_chunk_length_distribution.png)

| 항목 &emsp;&emsp;&emsp;&emsp; | 내용 |
| :--- | :--- |
| **구성** | 청크로 자른 3개 소스(회사 개요·고객 만족도·채용 가이드)의 청크 길이를 히스토그램으로 표시. 빨간 점선은 `chunk_size=500` 상한선 |
| **확인 포인트** | 회사 개요(파랑)는 480~500자에 청크가 가장 많이 몰려 있음 — Splitter가 상한 근처까지 빡빡하게 채우려 한다는 증거 / 고객 만족도(초록)는 130자 부근에 짧게 모임 — 한 줄 CSV가 그대로 청크가 되었기 때문 |
| **의미** | `chunk_size`는 "잘릴 수 있는 최대 길이"이며, 실제 청크 길이는 문서 형태에 따라 천차만별. 검색 품질을 튜닝할 때 이 분포를 먼저 확인해야 함 |

### 3. 질문 유형 → 호출되는 RAG Tool

![tool routing map](results/fig_04_tool_routing_map.png)

| 항목 &emsp;&emsp;&emsp;&emsp; | 내용 |
| :--- | :--- |
| **구성** | 8개 예시 질문(행) × 5개 검색 함수(열). 챗봇이 어떤 질문에 어떤 함수를 부르는지 매트릭스로 표시 |
| **확인 포인트** | "회사 슬로건"·"해외 거점" 같은 일반 질문은 모두 `company_profile`로 라우팅됨 / "GaN 모듈"·"CAN 프로토콜" 같은 구체 사양 질문은 `products_services`로 정확히 라우팅됨 |
| **의미** | 시스템 프롬프트에 적은 5개 함수 설명만으로도 LLM이 질문 주제를 파악해 한 함수만 호출함. 컬렉션을 분리한 효과를 직관적으로 확인 |

---

## ✨ 주요 결과 및 분석 (Key Findings & Analysis)

| 발견한 사실 &emsp;&emsp;&emsp; | 관찰 내용과 적용 방법 |
| :--- | :--- |
| **JSON은 자르면 안 된다** | 처음에는 모든 문서를 똑같이 청크로 자르려 했는데, JSON을 잘라 임베딩하니 "GaN 모듈 효율"을 물어도 엉뚱한 키만 검색되는 일이 생김. → 구조화 데이터(JSON/JSONL)는 키-값 관계 자체가 의미라서 **통째로 임베딩**해야 함. 텍스트와 같은 전략을 강제로 적용하면 오히려 검색 품질이 무너짐 |
| **컬렉션 분리 = 검색 잡음 감소** | 5종 문서를 한 컬렉션에 다 넣어 봤더니, 채용 질문에 회사 연혁 청크가 섞여 검색됨. 컬렉션을 5개로 나누고 함수도 5개로 분리하자 라우팅이 명확해짐. → "검색 단위" 자체를 설계 단계에서 분리하는 게 검색 정확도에 가장 큰 영향을 줌 |
| **함수 설명이 곧 라우팅 규칙** | 5개 함수의 docstring을 자세히 적자, 시스템 프롬프트만 보고도 LLM이 질문 주제에 맞는 함수 하나만 정확히 호출함. → docstring은 사람이 보라고 적는 메모가 아니라, **챗봇의 라우팅 규칙** 그 자체. 컬렉션을 잘 나누었어도 docstring이 부실하면 라우팅이 어긋남 |
| **임베딩과 LLM은 다른 서버로** | 처음에는 한 프록시 URL로 두 호출을 다 했는데, 임베딩이 LLM 응답을 막는 병목이 됨. URL을 두 개로 분리하자 두 호출이 독립적으로 처리됨. → 운영 환경을 가정한다면 처음부터 **임베딩 서버와 LLM 서버를 분리**해 두는 것이 확장성·교체 측면에서 유리 |

---

## 💡 회고록 (Retrospective)

이 프로젝트를 시작하기 전까지 RAG라는 단어는 저에게 다소 막연하게 다가왔습니다. "검색 + 생성"이라는 짧은 설명을 여러 번 봤지만, 그것이 실제 코드에서는 어떻게 흘러가는지 잘 그려지지 않았습니다. 직접 5종 문서를 로더에 흘려 보면서 처음 분명해진 사실은, RAG가 무슨 새로운 마법 같은 기술이 아니라 **"문서를 잘게 자르고, 비슷한 조각을 찾아, LLM에게 함께 넘겨 주는"** 아주 단순한 구조라는 점이었습니다. 화려한 이름에 비해 안에서 일어나는 일은 의외로 평범했고, 그래서 오히려 어디에 어떻게 끼워 넣어야 할지가 더 잘 보였습니다.

가장 의외였던 부분은 JSON과 JSONL을 청크로 자르면 안 된다는 점이었습니다. 처음에는 "모든 문서를 똑같이 500자로 잘라 임베딩하면 되겠지"라고 단순하게 생각했습니다. 그런데 막상 JSON을 그대로 잘라 보니, "GaN 모듈 효율"을 물었을 때 엉뚱하게 키 이름만 들어 있는 청크가 검색되는 일이 자주 생겼습니다. 곰곰이 생각해 보니 JSON은 키와 값이 묶여 있어야 의미가 살아 있는 데이터인데, 그것을 글자 수 기준으로 자르면 키만 남거나 값만 남는 조각이 만들어지는 것이었습니다. 그래서 `chunkable` 플래그를 만들어 "이 데이터는 통째로 임베딩한다"는 분기를 추가했고, 그 즉시 사양 질문의 검색 정확도가 크게 좋아졌습니다. 자료의 형태에 따라 처리 방식을 다르게 가져가야 한다는 것을 머리로만 알다가 코드로 직접 확인한 경험이었습니다.

또 하나 인상 깊었던 부분은 컬렉션을 어떻게 나눌지에 대한 고민이었습니다. 처음에는 "어차피 임베딩으로 비슷한 걸 찾으니 컬렉션은 하나면 충분하겠지"라고 생각했습니다. 그런데 실제로 한 컬렉션에 모든 문서를 넣고 채용 관련 질문을 던지니, 채용 청크와 회사 연혁 청크가 섞여 검색되는 문제가 생겼습니다. 단순히 잡음이 끼는 정도가 아니라, LLM이 그 잡음에 휘둘려 잘못된 답을 만들어 내기까지 했습니다. 컬렉션을 5개로 분리하고 함수도 5개로 만들자, 챗봇이 "이 질문은 채용 컬렉션만 보면 된다"는 결정을 먼저 내린 뒤 검색하기 때문에 잡음 자체가 들어올 여지가 사라졌습니다. 검색 품질을 튜닝하기 전에 **검색 단위 자체를 어떻게 나눌지** 가 더 큰 결정이라는 것을 직접 체감했습니다.

함수 설명(docstring)의 중요성은 이전 프로젝트에서도 느꼈는데, 이번에는 그게 단순한 사용 안내를 넘어 **라우팅 규칙** 역할을 한다는 점을 새롭게 알게 됐습니다. 5개 함수의 설명을 한두 문장 수준으로 짧게 적었더니, "GaN 모듈 효율"을 물어도 회사 개요 함수가 호출되는 어긋남이 자주 일어났습니다. 설명에 "구체적 제품 사양·기술 스펙 질문에 사용한다"는 식으로 호출 시점을 명확하게 적자, 라우팅이 거의 완벽해졌습니다. 컬렉션을 잘 나눈 것만으로는 부족하고, 그 컬렉션을 LLM에게 어떻게 소개할지가 함께 잘 설계되어야 한다는 점을 알게 됐습니다.

작은 발견이지만 임베딩 서버와 LLM 서버를 따로 두는 부분도 인상적이었습니다. 처음에는 두 호출을 같은 URL로 보냈는데, 임베딩 작업이 길어지면 LLM 응답까지 함께 느려지는 일이 있었습니다. 환경 변수를 `BASE_URL`과 `EMBEDDING_URL`로 분리하고 두 호출을 별도 프록시로 보내자 두 작업이 서로 영향을 주지 않게 됐습니다. 화려한 최적화는 아니지만, "운영 환경이라면 처음부터 두 서버를 분리해 두는 게 정석"이라는 감각을 얻은 것이 큰 수확이었습니다. 임베딩 모델을 나중에 다른 것으로 바꾸고 싶을 때도, 두 서버를 분리해 두면 그 부분만 갈아 끼우면 되기 때문에 확장성 측면에서도 유리했습니다.

이번 프로젝트를 통해 RAG의 진짜 힘은 모델이 아니라 **데이터를 어떻게 정리하고 나누느냐**에 달려 있다는 것을 직접 느꼈습니다. 문서 종류에 따라 로더를 다르게 쓰고, 구조에 따라 청크 전략을 분기하고, 검색 단위를 컬렉션으로 분리하는 일련의 결정들이 결국 챗봇 답변 품질을 좌우했습니다. 다음 단계에서는 검색된 문서를 단순히 그대로 LLM에 넘기는 게 아니라, 검색 결과를 한 번 더 정렬하거나(re-ranking), 여러 문서를 종합해 답을 만드는 더 정교한 RAG 구조로 나아가 보고 싶습니다. 이번에 만든 5종 문서 파이프라인이, 더 복잡한 검색 시스템을 시도할 때 든든한 출발점이 되어 줄 것이라 믿습니다.

---

## 🔗 참고 자료 (References)

- LangChain Documentation — Document Loaders / Text Splitters / Vector Stores
- ChromaDB — Open-source AI Application Database (Chroma, 2024)
- PyMuPDF — High-performance PDF parsing for Python (Artifex, 2024)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- LangChain Community — `RecursiveCharacterTextSplitter` API Reference
- NVIDIA AI ACADEMY · 챗봇 프로젝트 — `chapter_04_rag`
