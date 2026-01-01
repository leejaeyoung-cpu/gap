# 🏥 AI 기반 항암제 칵테일 최적화 시스템
## AI-Based Anticancer Cocktail Optimization System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

---

## 📋 목차 (Table of Contents)

1. [프로젝트 개요](#-프로젝트-개요)
2. [문제 상황](#-문제-상황)
3. [해결 방안](#-해결-방안)
4. [시스템 아키텍처](#-시스템-아키텍처)
5. [기술 스택](#-기술-스택)
6. [핵심 AI 기법](#-핵심-ai-기법)
7. [데이터 파이프라인](#-데이터-파이프라인)
8. [주요 기능](#-주요-기능)
9. [설치 및 실행](#-설치-및-실행)
10. [기대 효과](#-기대-효과)
11. [로드맵](#-로드맵)
12. [라이선스](#-라이선스)

---

## 🎯 프로젝트 개요

### 서비스명
**GAP (Genetic Anticancer Personalization)** - AI 기반 개인맞춤형 항암제 칵테일 최적화 시스템

### 핵심 가치 제안
```
환자 데이터 입력 → AI 분석 (1-2분) → 최적 항암제 칵테일 제시
✓ 최소 농도로 독성 최소화
✓ 최적 시그널 패스웨이 차단
✓ 안정적인 암 사멸 유도
```

### 타겟 사용자
- **1차**: 종양내과 전문의 (항암제 처방 의사결정 지원)
- **2차**: 임상연구자 (신규 항암제 조합 연구)
- **3차**: 병원 약제부 (약물 상호작용 검증)

### 개발 목표
암 환자 개개인의 종양 특성, 유전체 정보, 약물 대사 능력을 분석하여 **최소 독성·최대 효능**의 항암제 조합과 용량을 AI로 자동 추천하는 의사결정 지원 시스템 구축

---

## 🔴 문제 상황

### 1.1 현재 항암 치료의 Pain Points

#### 표준 치료 프로토콜의 한계
```
문제점:
├─ 획일적 용량 (체표면적 기반)
│   → 개인 대사 차이 무시
│   → 과다/과소 투여 위험
│
├─ 제한적 약물 조합
│   → 임상시험 기반 표준 요법만 사용
│   → 신규 조합 탐색 어려움
│
├─ 시행착오 방식
│   → 1차 치료 실패 → 2차 치료 시도
│   → 환자 고통 증가, 치료 기회 상실
│
└─ 독성 예측 불가
    → 부작용 발생 후 대응
    → 치료 중단/용량 감량
```

#### 의료진의 어려움
- **복잡한 약물 상호작용**: 3개 이상 병용 시 상호작용 예측 곤란
- **제한된 정보**: 개별 환자의 약물 반응성 데이터 부족
- **시간 부족**: 환자당 진료 시간 제한 (평균 5-10분)
- **의사결정 부담**: 생명과 직결된 고위험 의사결정

### 1.2 기존 솔루션의 한계

#### 전통적 임상시험 기반 접근
```
한계:
├─ 시간: 신약 조합 검증에 5-10년 소요
├─ 비용: 임상시험 1건당 수백억 원
├─ 범위: 제한된 조합만 검증 가능
└─ 일반화: 평균 환자 기준, 개인 맞춤 불가
```

#### 기존 AI 솔루션
```
문제점:
├─ IBM Watson for Oncology
│   → 2022년 서비스 중단
│   → 실제 임상 환경 적용 실패
│   → 권고안의 낮은 정확도 (30-40%)
│
└─ 학술 연구 모델
    → 실험실 수준 (in vitro/in silico)
    → 실제 환자 데이터 부족
    → 임상 적용 불가
```

### 1.3 시장 기회

#### 시장 규모
```
글로벌 항암제 시장:
├─ 2023년: 1,960억 달러
├─ 2030년: 3,770억 달러 (예상)
└─ CAGR: 9.8%

한국 항암제 시장:
├─ 2023년: 3조 5천억 원
├─ 연평균 성장률: 12.3%
└─ 바이오시밀러 포함 시 5조 원+
```

#### 규제 환경
```
유리한 요소:
├─ 식약처: AI 의료기기 허가 가이드라인 마련 (2021)
├─ 건강보험: 정밀의료 급여 확대
├─ 병원: EMR(전자의무기록) 데이터 축적
└─ 정부: K-Bio 육성 정책, R&D 지원 확대
```

---

## 💡 해결 방안

### 2.1 핵심 컨셉

```
┌──────────────────────────────────────────────────────────┐
│  입력 (Multi-modal Data)                                  │
│  ├─ 세포 이미지 (H&E, IHC, 형광 현미경)                  │
│  ├─ 유전체 정보 (NGS, RNA-seq)                           │
│  ├─ 임상 데이터 (나이, 간/신 기능, 병력)                 │
│  └─ 종양 특성 (암종, 병기, 바이오마커)                   │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  AI 처리 엔진                                             │
│  ├─ Cellpose: 세포 분할 및 형태 분석                     │
│  ├─ CNN/Transformer: 이미지 특징 추출                    │
│  ├─ GNN: 약물-단백질 상호작용 분석                       │
│  ├─ Dose-Response 모델: 용량 최적화                      │
│  └─ 앙상블 모델: 독성 예측                               │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  출력 (Personalized Recommendation)                       │
│  ├─ 최적 약물 조합 (2-4개 항암제)                        │
│  ├─ 개인맞춤 용량 (mg/m² 또는 mg)                        │
│  ├─ 투여 스케줄 (Day 1, 8, 15 등)                        │
│  ├─ 예상 효능 (반응률 %)                                 │
│  ├─ 예상 독성 (Grade 0-4)                                │
│  └─ 대안 요법 (2nd, 3rd line)                            │
└──────────────────────────────────────────────────────────┘
```

### 2.2 주요 기능

#### 🔬 1. 자동 세포 이미지 분석
```
Cellpose AI 기반:
├─ 암세포 자동 분할 (Segmentation)
├─ 형태학적 특징 추출 (500+ features)
├─ 핵-세포질 비율 (N/C ratio) 계산
├─ 세포 밀도 및 공간 분포 분석
└─ 바이오마커 정량화 (HER2, PD-L1 등)

처리 속도:
├─ GPU: 2-5초/이미지 (1024×1024)
└─ CPU: 20-30초/이미지
```

#### 💊 2. 항암제 칵테일 최적화
```
AI 추론 엔진:
├─ 540만+ 약물 조합 시뮬레이션
├─ Synergy Score 계산 (Bliss, Loewe 모델)
├─ 신호 패스웨이 커버리지 분석
├─ 내성 메커니즘 고려
└─ 비용-효과 분석

출력:
├─ Top 5 약물 조합 (우선순위)
├─ 각 조합별 예상 반응률
├─ 상승 효과 (Synergy) 점수
└─ 근거 논문 (PubMed 링크)
```

#### ⚖️ 3. 개인맞춤 용량 계산
```
다층 알고리즘:
├─ BSA 기반 표준 용량 계산
├─ 약물 상호작용 분석 (DDI)
├─ 간/신 기능 보정
├─ 유전자형 고려 (CYP450 등)
├─ 독성 위험 예측
└─ Dose-Response 곡선 최적화

안전 장치:
├─ 최대 용량 제한 (cap)
├─ 단계적 증량 프로토콜
└─ Red flag 알림 (위험 조합)
```

#### 📊 4. 실시간 학습 시스템
```
Continuous Learning:
├─ 신규 환자 데이터 자동 수집
├─ 치료 결과 추적 (3개월, 6개월)
├─ 모델 성능 모니터링
├─ 주기적 재학습 (월 1회)
└─ A/B 테스트 (AI vs 기존 요법)

버전 관리:
├─ 모델 v1.0 → v1.1 → v2.0
├─ 성능 향상 기록
└─ 롤백 가능
```

### 2.3 차별화 포인트

| 기존 방법 | 본 시스템 |
|----------|----------|
| 표준 프로토콜 (획일적) | AI 개인 맞춤형 |
| 2-3개월 치료 후 효과 평가 | 1-2분 내 예측 제공 |
| 제한적 약물 조합 (10여 개) | 540만+ 조합 분석 |
| 주관적 의사결정 | 데이터 기반 근거 제시 |
| 독성 사후 대응 | 사전 예측 및 예방 |
| 고정 용량 | 동적 용량 조정 |

---

## 🏗️ 시스템 아키텍처

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Streamlit   │  │   Gradio     │  │  Web Portal  │      │
│  │  Dashboard   │  │   Interface  │  │  (병원 EMR)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTPS/REST API
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                   │
│  ├─ Authentication (JWT)                                     │
│  ├─ Rate Limiting                                            │
│  ├─ Request Validation                                       │
│  └─ Load Balancing                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Image Analysis  │  │ Drug Optimizer  │                  │
│  │    Service      │  │    Service      │                  │
│  │  - Cellpose     │  │  - Dose Calc    │                  │
│  │  - Feature      │  │  - DDI Check    │                  │
│  │    Extraction   │  │  - Synergy      │                  │
│  └─────────────────┘  └─────────────────┘                  │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  ML Inference   │  │ Learning Engine │                  │
│  │    Engine       │  │                 │                  │
│  │  - TensorFlow   │  │  - Model Update │                  │
│  │  - PyTorch      │  │  - Validation   │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │    MongoDB   │  │     Redis    │     │
│  │  (구조화 데이터)│ │  (이미지/문서)│ │   (캐시)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │      S3      │  │  Vector DB   │                        │
│  │  (파일 저장) │  │  (Pinecone)  │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Docker    │  │  Kubernetes  │  │   AWS/GCP    │     │
│  │  Containers  │  │ Orchestration│  │     Cloud    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 흐름 (Data Flow)

```
1. 데이터 수집
   ├─ 이미지 업로드 (PNG/TIFF) → S3 저장
   ├─ 환자 정보 입력 → PostgreSQL
   └─ 유전체 데이터 → MongoDB

2. 전처리
   ├─ 이미지 정규화 (Normalization)
   ├─ 결측치 처리 (Imputation)
   └─ Feature Engineering

3. AI 추론
   ├─ GPU 클러스터 로드 밸런싱
   ├─ 배치 처리 (Batch Inference)
   └─ 결과 캐싱 (Redis)

4. 결과 생성
   ├─ 보고서 PDF 생성
   ├─ 시각화 차트
   └─ EMR 연동 (HL7/FHIR)

5. 피드백 루프
   ├─ 치료 결과 수집
   ├─ 모델 성능 평가
   └─ 재학습 트리거
```

---

## 🛠️ 기술 스택

### 4.1 Frontend

```yaml
Primary:
  - Streamlit: 3.8+
    역할: 빠른 프로토타이핑 및 연구용 대시보드
    장점: Python 네이티브, 데이터 과학 친화적
    
  - Gradio: 4.0+
    역할: 이미지 업로드 및 인터랙티브 UI
    장점: 간편한 ML 모델 배포

Alternative (프로덕션):
  - React.js: 18.x
  - Next.js: 14.x
  - TypeScript: 5.x
  
Visualization:
  - Plotly: 인터랙티브 차트
  - Matplotlib / Seaborn: 정적 그래프
  - Altair: 선언형 시각화
```

### 4.2 Backend

```yaml
API Framework:
  - FastAPI: 0.110+
    이유: 
      - 빠른 성능 (Starlette 기반)
      - 자동 API 문서 (Swagger/OpenAPI)
      - 비동기 처리 (async/await)
      - Pydantic 타입 검증
      
Task Queue:
  - Celery: 5.3+
    역할: 장시간 작업 비동기 처리
    - 이미지 배치 분석
    - 모델 재학습
    - 대량 시뮬레이션
  
  - Redis: 7.2+
    역할: 
      - Celery 메시지 브로커
      - 결과 캐싱
      - 세션 관리

Web Server:
  - Uvicorn: ASGI 서버
  - Gunicorn: 프로세스 관리자
  - Nginx: 리버스 프록시, 정적 파일
```

### 4.3 AI/ML Stack

```yaml
Deep Learning:
  Primary:
    - TensorFlow: 2.15+
    - Keras: 3.0+
  Secondary:
    - PyTorch: 2.1+ (연구용)
  
Computer Vision:
  - Cellpose: 3.0+
    핵심 알고리즘, 세포 분할
  - OpenCV: 4.9+
    이미지 전처리
  - scikit-image: 0.22+
    형태학적 분석
    
Graph Neural Network:
  - PyTorch Geometric: 2.4+
    약물-단백질 상호작용 그래프
  - NetworkX: 3.2+
    시그널 패스웨이 분석
    
Traditional ML:
  - scikit-learn: 1.4+
    - Random Forest (독성 예측)
    - XGBoost (반응률 예측)
    - SHAP (설명 가능 AI)
    
NLP (문헌 분석):
  - Transformers: 4.36+
    BioBERT, PubMedBERT
  - LangChain: 0.1+
    논문 요약 및 QA
    
Optimization:
  - SciPy: 1.12+
    용량 최적화
  - CVXPY: 1.4+
    볼록 최적화
```

### 4.4 Data & Database

```yaml
Relational DB:
  - PostgreSQL: 16+
    데이터:
      - 환자 기본 정보
      - 치료 이력
      - 검사 결과
      - 사용자 관리
    
NoSQL:
  - MongoDB: 7.0+
    데이터:
      - 이미지 메타데이터
      - 유전체 Raw 데이터
      - 비정형 임상 기록
      
Vector Database:
  - Pinecone / Weaviate
    데이터:
      - 논문 임베딩
      - 약물 구조 벡터
      - 유사 환자 검색
      
Object Storage:
  - AWS S3 / MinIO
    데이터:
      - 원본 이미지 (TIFF, PNG)
      - 모델 체크포인트
      - 백업 파일
      
Data Processing:
  - Pandas: 2.2+
  - NumPy: 1.26+
  - Polars: 0.20+ (대용량 데이터)
```

### 4.5 Infrastructure & DevOps

```yaml
Containerization:
  - Docker: 24.0+
  - Docker Compose: 2.24+
  
Orchestration:
  - Kubernetes: 1.29+
  - Helm: 3.14+
  
Cloud Platform (선택):
  Option A (AWS):
    - EC2: P3/P4 인스턴스 (GPU)
    - EKS: Kubernetes 관리
    - S3: 스토리지
    - RDS: PostgreSQL 관리형
    - CloudWatch: 모니터링
    
  Option B (GCP):
    - Compute Engine: GPU 인스턴스
    - GKE: Kubernetes
    - Cloud Storage
    - Cloud SQL
    
  Option C (On-Premise):
    - 병원 내부 서버
    - NVIDIA DGX 또는 A100 GPU 서버
    
CI/CD:
  - GitHub Actions
  - GitLab CI
  - ArgoCD (GitOps)
  
Monitoring:
  - Prometheus: 메트릭 수집
  - Grafana: 시각화
  - ELK Stack: 로그 분석
    - Elasticsearch
    - Logstash
    - Kibana
```

### 4.6 Security & Compliance

```yaml
Authentication:
  - JWT (JSON Web Tokens)
  - OAuth 2.0 / OIDC
  
Encryption:
  - TLS 1.3 (전송 중)
  - AES-256 (저장 시)
  
Compliance:
  - HIPAA (미국 의료정보보호법)
  - GDPR (EU 개인정보보호법)
  - 생명윤리법 (한국)
  - 개인정보보호법 (한국)
  
Audit:
  - 모든 접근 로그 기록
  - 데이터 변경 이력 추적
  - 정기 보안 감사
```

### 4.7 External APIs & Services

```yaml
Biomedical Databases:
  - PubMed API
    논문 검색 및 다운로드
  - DrugBank API
    약물 정보 조회
  - ClinicalTrials.gov API
    임상시험 데이터
  - TCGA (The Cancer Genome Atlas)
    유전체 데이터
    
Cloud AI Services (Optional):
  - Google Cloud Vision API
    이미지 품질 검증
  - AWS SageMaker
    모델 학습 가속
```

---

## 🧠 핵심 AI 기법

### 5.1 Cellpose: 세포 이미지 분할

#### 원리
```
Gradient Vector Flow (경사 벡터 흐름):

각 픽셀이 세포 중심을 가리키는 벡터를 CNN으로 예측
→ 동역학 시뮬레이션으로 픽셀을 중심으로 이동
→ 동일 중심에 수렴한 픽셀 = 하나의 세포
```

#### 네트워크 구조
```
Input Image (H×W×C)
    ↓
┌─────────────────────────┐
│   Encoder (U-Net 기반)   │
│   Conv → BatchNorm → ReLU│
│   MaxPool (4단계)        │
│   64 → 128 → 256 → 512  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   Decoder (Upsampling)   │
│   UpConv + Skip Connect  │
│   Concatenate            │
└─────────────────────────┘
    ↓
Output:
├─ dX (X방향 벡터)
├─ dY (Y방향 벡터)
└─ Cell Probability (세포 확률)
```

#### 성능
```
사전 학습 모델:
├─ cyto (세포질): F1-score 0.87
└─ nuclei (핵): F1-score 0.92

본 시스템 Fine-tuning 후:
└─ 암세포 이미지: F1-score 0.90+

처리 속도:
├─ GPU (NVIDIA A100): 2-3초 / 1024×1024 이미지
└─ CPU: 25-30초
```

### 5.2 Multi-Modal Fusion (멀티모달 융합)

```python
# 개념적 코드
class MultiModalModel(nn.Module):
    def __init__(self):
        # 이미지 브랜치
        self.image_encoder = EfficientNetB3()
        
        # 유전체 브랜치
        self.genomic_encoder = TransformerEncoder()
        
        # 임상 데이터 브랜치
        self.clinical_encoder = MLP()
        
        # 융합 레이어
        self.fusion = AttentionFusion()
        
    def forward(self, image, genomic, clinical):
        img_feat = self.image_encoder(image)      # → 1024-dim
        gen_feat = self.genomic_encoder(genomic)  # → 512-dim
        cli_feat = self.clinical_encoder(clinical) # → 256-dim
        
        # Cross-Attention Fusion
        fused = self.fusion([img_feat, gen_feat, cli_feat])
        
        # 출력 헤드
        drug_prediction = self.drug_head(fused)
        dose_prediction = self.dose_head(fused)
        toxicity_prediction = self.toxicity_head(fused)
        
        return drug_prediction, dose_prediction, toxicity_prediction
```

### 5.3 Graph Neural Network (약물 상호작용)

```
약물-단백질 상호작용 그래프:

Nodes:
├─ 약물 (Drug)
├─ 단백질 (Protein/Target)
└─ 패스웨이 (Pathway)

Edges:
├─ 약물 → 단백질 (억제/활성화)
├─ 단백질 → 패스웨이
└─ 약물 ↔ 약물 (상호작용)

GNN 처리:
├─ Message Passing (3 layers)
├─ Graph Attention (GAT)
├─ Readout (Global Pooling)
└─ Synergy Score 예측
```

### 5.4 Dose-Response Modeling

```
Hill Equation:
E = E_max × D^n / (EC50^n + D^n)

여기서:
- E: 효과 (0-1)
- D: 용량
- EC50: 50% 효과 용량
- n: Hill coefficient

최적화 목표:
Maximize: E(D_combo) - E(D_A) - E(D_B)  (상승 효과)
Subject to:
  - Toxicity(D_combo) < Threshold
  - D_A, D_B > 0
```

### 5.5 설명 가능 AI (Explainable AI)

```
SHAP (SHapley Additive exPlanations):

각 입력 특징이 예측에 기여한 정도를 계산

예시 출력:
┌─────────────────────────────────────────┐
│ 약물 A (Doxorubicin) 추천 근거:         │
│ ├─ HER2 발현 높음: +0.23              │
│ ├─ 종양 크기 큼: +0.18                │
│ ├─ ER/PR 음성: +0.15                  │
│ ├─ 젊은 나이: +0.08                   │
│ └─ LVEF 정상: +0.05                   │
│                                          │
│ 반대 요인:                               │
│ └─ 이전 안트라사이클린 사용: -0.12      │
└─────────────────────────────────────────┘
```

---

## 📊 데이터 파이프라인

### 6.1 데이터 소스

```yaml
1차 데이터 (직접 수집):
  - 환자 세포 이미지:
      형식: TIFF, PNG (1024×1024 이상)
      염색: H&E, IHC (HER2, ER, PR, Ki67)
      수량: 5,000+ 이미지 (목표)
      
  - 환자 임상 정보:
      인구통계: 나이, 성별, BMI
      검사 결과: CBC, LFT, RFT
      병리 정보: 암종, 병기, 분화도
      
  - 치료 데이터:
      처방 약물 및 용량
      투여 스케줄
      치료 반응 (RECIST)
      부작용 (CTCAE Grade)

2차 데이터 (공개 데이터베이스):
  - TCGA (The Cancer Genome Atlas):
      유전체 데이터
      생존 분석 데이터
      
  - GDSC (Genomics of Drug Sensitivity):
      약물 반응성 데이터
      IC50 값
      
  - DrugBank:
      약물 구조
      약동학/약력학 정보
      
  - PubMed:
      항암제 조합 연구 논문
      임상시험 결과

3차 데이터 (외부 협력):
  - 병원 EMR 연동 (IRB 승인 후)
  - 제약사 임상시험 데이터 (협약)
```

### 6.2 데이터 전처리

```python
# 이미지 전처리 파이프라인
def preprocess_image(image_path):
    """
    세포 이미지 전처리
    """
    # 1. 로드
    img = tifffile.imread(image_path)
    
    # 2. 색상 정규화 (Reinhard)
    img_normalized = normalize_staining(img, method='reinhard')
    
    # 3. 타일 분할 (큰 이미지)
    if img.shape[0] > 2048 or img.shape[1] > 2048:
        tiles = split_into_tiles(img, tile_size=1024, overlap=128)
    else:
        tiles = [img]
    
    # 4. 품질 확인
    tiles = [t for t in tiles if check_quality(t, min_sharpness=100)]
    
    # 5. 증강 (Augmentation) - 학습 시에만
    if mode == 'train':
        tiles = augment_images(tiles, 
                               rotation=True,
                               flip=True,
                               brightness=0.1,
                               contrast=0.1)
    
    return tiles

# 임상 데이터 전처리
def preprocess_clinical(df):
    """
    임상 데이터 전처리
    """
    # 1. 결측치 처리
    df['creatinine'].fillna(df['creatinine'].median(), inplace=True)
    
    # 2. 이상치 제거 (IQR)
    df = remove_outliers(df, columns=['age', 'bmi'])
    
    # 3. 범주형 변수 인코딩
    df = pd.get_dummies(df, columns=['cancer_type', 'stage'])
    
    # 4. 정규화
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
```

### 6.3 ERD (Entity Relationship Diagram)

```
┌─────────────────┐       ┌─────────────────┐
│    Patients     │       │  CellImages     │
├─────────────────┤       ├─────────────────┤
│ patient_id (PK) │───┬───│ image_id (PK)   │
│ name            │   │   │ patient_id (FK) │
│ age             │   │   │ file_path       │
│ gender          │   │   │ staining_type   │
│ bmi             │   │   │ upload_date     │
│ cancer_type     │   │   │ analyzed        │
└─────────────────┘   │   └─────────────────┘
                      │   
                      │   ┌─────────────────┐
                      ├───│ CellFeatures    │
                      │   ├─────────────────┤
                      │   │ feature_id (PK) │
                      │   │ image_id (FK)   │
                      │   │ cell_count      │
                      │   │ mean_area       │
                      │   │ nc_ratio        │
                      │   │ ... (500+ cols) │
                      │   └─────────────────┘
                      │   
                      │   ┌─────────────────┐
                      └───│  Treatments     │
                          ├─────────────────┤
                          │ treatment_id(PK)│
                          │ patient_id (FK) │
                          │ drug_combo      │
                          │ dosage          │
                          │ start_date      │
                          │ response (RECIST)│
                          │ toxicity (CTCAE)│
                          └─────────────────┘
                              ↓
                      ┌─────────────────┐
                      │  DrugInteractions│
                      ├─────────────────┤
                      │ interaction_id  │
                      │ drug_a          │
                      │ drug_b          │
                      │ severity        │
                      │ mechanism       │
                      └─────────────────┘
```

---

## ⚙️ 주요 기능

### 7.1 환자 등록 및 데이터 입력

```
사용자 인터페이스:
┌──────────────────────────────────────┐
│  환자 정보 입력                      │
│  ├─ 기본 정보: ID, 이름, 나이, 성별 │
│  ├─ 신체 계측: 키, 체중 (→ BSA)    │
│  ├─ 암 정보: 암종, 병기, 조직형     │
│  └─ 검사 결과: 간/신 기능, 혈액     │
└──────────────────────────────────────┘
        ↓
┌──────────────────────────────────────┐
│  이미지 업로드                       │
│  ├─ Drag & Drop 또는 파일 선택     │
│  ├─ 지원 형식: TIFF, PNG, JPG      │
│  ├─ 배치 업로드 가능 (최대 50장)   │
│  └─ 자동 품질 검증                  │
└──────────────────────────────────────┘
        ↓
┌──────────────────────────────────────┐
│  유전체 정보 (선택)                  │
│  ├─ NGS 파일 업로드 (VCF)          │
│  ├─ 또는 주요 변이만 입력:          │
│  │   EGFR, KRAS, BRAF, HER2 등     │
│  └─ TMB, MSI 상태                   │
└──────────────────────────────────────┘
```

### 7.2 AI 분석 실행

```
진행 상황 표시:
┌──────────────────────────────────────┐
│  분석 진행 중...                     │
│  ━━━━━━━━━━━━━━━━━━━━━━ 75%       │
│                                       │
│  ✓ 이미지 전처리 완료 (10초)        │
│  ✓ Cellpose 분석 완료 (45초)        │
│  ✓ 특징 추출 완료 (15초)            │
│  ⏳ 약물 조합 최적화 중... (30초)   │
│  ⏹ 보고서 생성 대기 중              │
└──────────────────────────────────────┘

총 예상 시간: 약 2분
```

### 7.3 결과 보고서

```
┌─────────────────────────────────────────────────┐
│  환자 ID: P-2024-001 | 분석일: 2024-12-30     │
├─────────────────────────────────────────────────┤
│  【종양 특성 분석】                             │
│  ├─ 세포 수: 3,482개 검출                      │
│  ├─ 평균 세포 크기: 245 μm²                    │
│  ├─ N/C 비율: 0.58 (높음, 악성 시사)          │
│  ├─ 세포 밀도: 1,240 cells/mm²                │
│  └─ HER2 Score: 2+ (IHC)                       │
├─────────────────────────────────────────────────┤
│  【권장 항암제 조합】                           │
│                                                  │
│  🥇 1순위 (추천도 92%)                         │
│  ┌───────────────────────────────────┐        │
│  │ Trastuzumab + Pertuzumab + Docetaxel│        │
│  ├───────────────────────────────────┤        │
│  │ 예상 반응률: 78% (ORR)              │        │
│  │ 예상 독성:                          │        │
│  │  ├─ 골수억제: Grade 2 (중등도)      │        │
│  │  ├─ 심독성: Grade 1 (경미)         │        │
│  │  └─ 설사: Grade 1                  │        │
│  │                                     │        │
│  │ 용량:                               │        │
│  │  ├─ Trastuzumab: 8 mg/kg (1일차)   │        │
│  │  │              6 mg/kg (이후)      │        │
│  │  ├─ Pertuzumab: 840 mg (1일차)     │        │
│  │  │              420 mg (이후)       │        │
│  │  └─ Docetaxel: 75 mg/m² = 137 mg   │        │
│  │                                     │        │
│  │ 스케줄: 3주마다 반복 (6 사이클)     │        │
│  │                                     │        │
│  │ 근거 논문: 5편 (PubMed 링크)        │        │
│  └───────────────────────────────────┘        │
│                                                  │
│  🥈 2순위 (추천도 85%)                         │
│  T-DM1 (Trastuzumab emtansine) 단독...         │
│                                                  │
│  🥉 3순위 (추천도 79%)                         │
│  Lapatinib + Capecitabine...                    │
├─────────────────────────────────────────────────┤
│  【주의사항】                                   │
│  ⚠ LVEF 정기 모니터링 필요 (심독성 예방)       │
│  ⚠ G-CSF 예방적 투여 고려 (호중구감소증)       │
│  📋 간/신 기능 검사: 매 사이클마다              │
└─────────────────────────────────────────────────┘
```

### 7.4 시각화 대시보드

```
Dashboard Panels:

1. 세포 분할 결과
   ├─ 원본 이미지
   ├─ 마스크 오버레이
   └─ 각 세포별 컬러 맵

2. 특징 분포
   ├─ 세포 크기 히스토그램
   ├─ N/C 비율 분포
   └─ 바이오마커 발현 차트

3. 약물 조합 비교
   ├─ 효능 vs 독성 산점도
   ├─ 비용 비교 막대 그래프
   └─ Pareto Frontier

4. 신호 패스웨이
   ├─ 약물-표적 네트워크
   ├─ 차단 경로 하이라이트
   └─ 상승 효과 시각화

5. 생존 곡선 (예측)
   ├─ Kaplan-Meier 곡선
   ├─ 신뢰 구간
   └─ 중앙 생존 기간
```

### 7.5 지속적 학습 및 버전 관리

```
자동 학습 파이프라인:

1. 데이터 수집 (매주)
   ├─ 신규 환자 50명
   └─ 치료 결과 수집

2. 데이터 검증 (자동)
   ├─ 품질 체크
   ├─ 이상치 탐지
   └─ 레이블 검증

3. 모델 재학습 (월 1회)
   ├─ 기존 데이터 + 신규 데이터
   ├─ Validation Set 분리
   └─ Cross-validation

4. 성능 평가
   ├─ Accuracy, Precision, Recall
   ├─ AUC-ROC
   └─ 기존 모델과 비교

5. A/B 테스트 (2주)
   ├─ 신규 모델 20% 트래픽
   ├─ 기존 모델 80%
   └─ 실시간 성능 모니터링

6. 배포 결정
   IF 성능 향상 > 2% AND 안정성 OK:
       → 신규 모델 100% 배포
   ELSE:
       → 기존 모델 유지, 원인 분석
```

---

## 🚀 설치 및 실행

### 8.1 시스템 요구사항

```yaml
최소 사양 (개발/테스트):
  CPU: Intel i5 이상 / AMD Ryzen 5 이상
  RAM: 16GB
  GPU: NVIDIA GTX 1660 이상 (6GB VRAM)
  Storage: SSD 100GB 이상

권장 사양 (프로덕션):
  CPU: Intel Xeon / AMD EPYC (16+ cores)
  RAM: 64GB 이상
  GPU: NVIDIA A100 (40GB) 또는 RTX 4090 (24GB)
  Storage: NVMe SSD 500GB 이상

소프트웨어:
  OS: Ubuntu 22.04 LTS / Windows 11 / macOS 13+
  Python: 3.8-3.11
  CUDA: 11.8 / 12.1 (GPU 사용 시)
  Docker: 24.0+ (선택)
```

### 8.2 설치 방법

#### Option A: Docker (권장)

```bash
# 1. 리포지토리 클론
git clone https://github.com/leejaeyoung-cpu/gap.git
cd gap

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일 편집 (데이터베이스 정보, API 키 등)

# 3. Docker Compose로 실행
docker-compose up -d

# 4. 서비스 확인
# Frontend: http://localhost:8501
# API: http://localhost:8000/docs
# Monitoring: http://localhost:3000 (Grafana)
```

#### Option B: Manual Setup

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 데이터베이스 초기화
python scripts/init_db.py

# 4. Cellpose 모델 다운로드
python -c "from cellpose import models; models.Cellpose(model_type='cyto3')"

# 5. 사전 학습 모델 다운로드
python scripts/download_models.py

# 6. 백엔드 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &

# 7. 프론트엔드 실행
streamlit run app/frontend/dashboard.py --server.port 8501
```

### 8.3 초기 설정

```bash
# 관리자 계정 생성
python scripts/create_admin.py \
  --username admin \
  --email admin@hospital.com \
  --password [secure_password]

# 샘플 데이터 로드 (선택)
python scripts/load_sample_data.py

# GPU 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# 출력 예: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 8.4 기본 사용법

```python
# Python API 사용 예시
from gap import GAP

# 클라이언트 초기화
client = GAP(api_key='your_api_key')

# 환자 등록
patient = client.create_patient(
    name='홍길동',
    age=55,
    gender='F',
    weight_kg=62,
    height_cm=158,
    cancer_type='Breast',
    stage='IIB'
)

# 이미지 분석
analysis = client.analyze_image(
    patient_id=patient.id,
    image_path='path/to/cell_image.tiff',
    staining='HER2'
)

# 항암제 추천
recommendation = client.recommend_drugs(
    patient_id=patient.id,
    n_recommendations=3
)

# 결과 출력
for i, rec in enumerate(recommendation, 1):
    print(f"{i}순위: {rec['drugs']}")
    print(f"  예상 반응률: {rec['response_rate']:.1%}")
    print(f"  독성 점수: {rec['toxicity_score']:.2f}")
    print()
```

---

## 📈 기대 효과

### 9.1 환자 관점

```
✓ 생존율 향상
  - 최적 약물 조합으로 15-20% 반응률 향상 (예상)
  
✓ 부작용 감소
  - 개인맞춤 용량으로 독성 30% 감소 (목표)
  
✓ 치료 기간 단축
  - 1차 치료 성공률 증가로 평균 치료 기간 감소
  
✓ 삶의 질 개선
  - 부작용 최소화로 일상 생활 유지 가능
```

### 9.2 의료진 관점

```
✓ 의사결정 지원
  - 데이터 기반 근거로 확신 있는 처방
  - 소송 리스크 감소
  
✓ 시간 절약
  - 문헌 검색 시간 90% 단축 (30분 → 3분)
  
✓ 지식 공유
  - 최신 치료 가이드라인 자동 업데이트
  - 경험이 적은 의사도 전문가 수준 처방 가능
```

### 9.3 병원 관점

```
✓ 비용 절감
  - 불필요한 약물 사용 감소
  - 부작용 관리 비용 감소
  
✓ 치료 성과 향상
  - 병원 평가 지표 개선
  - 환자 만족도 증가
  
✓ 연구 역량 강화
  - 축적된 데이터로 임상연구 수행
  - 논문 발표 및 특허 출원
```

### 9.4 사회경제적 영향

```
✓ 의료비 절감
  - 연간 약 2,000억 원 절감 가능 (전국 확대 시)
  
✓ 산업 육성
  - AI 헬스케어 생태계 활성화
  - 글로벌 수출 가능 (의료 AI 선도)
  
✓ 국민 건강 증진
  - 암 생존율 향상
  - 건강 수명 연장
```

---

## 🗺️ 로드맵

### Phase 1: MVP 개발 (3개월)
```
Month 1-2:
├─ 핵심 기능 구현
│   ├─ Cellpose 통합
│   ├─ 기본 UI (Streamlit)
│   └─ PostgreSQL 데이터베이스
│
└─ 초기 데이터 수집 (500 환자)

Month 3:
├─ 베타 테스트 (1개 병원)
├─ 피드백 수집 및 개선
└─ 성능 평가 보고서 작성
```

### Phase 2: 임상 검증 (6개월)
```
Month 4-6:
├─ IRB 승인 획득
├─ 전향적 임상연구 시작
├─ 모델 정확도 개선 (90%+ 목표)
└─ 3개 병원 확대 (500 → 2,000 환자)

Month 7-9:
├─ 중간 분석 및 논문 작성
├─ 식약처 의료기기 인증 준비
└─ 보험 급여 신청
```

### Phase 3: 상용화 (6개월)
```
Month 10-12:
├─ 의료기기 인증 취득 (3등급)
├─ 클라우드 배포 (AWS/GCP)
├─ 10개 병원 확대
└─ SaaS 모델 출시

Month 13-15:
├─ 전국 확산 (50개 병원)
├─ 글로벌 라이선스 (미국, 일본)
└─ Series A 투자 유치
```

### Phase 4: 고도화 (1년+)
```
Year 2:
├─ 멀티암종 확대
│   (유방암 → 대장암, 폐암, 위암)
├─ Real-World Evidence 구축
├─ 신약 개발 지원 기능 추가
└─ AI 자동 모니터링 시스템
```

---

## 📄 라이선스 및 윤리

### License
```
MIT License (오픈소스)

단, 의료 데이터는 엄격한 보안 규정 준수:
- HIPAA (미국)
- GDPR (EU)
- 개인정보보호법, 생명윤리법 (한국)
```

### 윤리적 고려사항
```
1. 투명성
   - AI 의사결정 과정 설명 가능
   - 한계 명확히 고지
   
2. 공정성
   - 특정 인종/성별 편향 제거
   - 다양한 환자군 포함
   
3. 책임성
   - 최종 결정은 의료진이 수행
   - AI는 보조 도구로만 사용
   
4. 개인정보 보호
   - 데이터 익명화
   - 접근 권한 엄격 관리
```

---

## 🤝 기여 및 문의

### 개발팀
```
Project Lead: 이재영
Email: lee.jaeyoung@gap-ai.com
GitHub: @leejaeyoung-cpu

기여자 환영:
- Issue 제기
- Pull Request
- 피드백 및 제안
```

### Citation
```bibtex
@software{gap2024,
  author = {Lee, Jaeyoung and Team},
  title = {GAP: AI-Based Anticancer Cocktail Optimization System},
  year = {2024},
  url = {https://github.com/leejaeyoung-cpu/gap}
}
```

---

## 📚 참고 문헌

1. Stringer, C., et al. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.

2. Preuer, K., et al. (2018). DeepSynergy: predicting anti-cancer drug synergy with Deep Learning. *Bioinformatics*, 34(9), 1538-1546.

3. Zitnik, M., et al. (2018). Modeling polypharmacy side effects with graph convolutional networks. *Bioinformatics*, 34(13), i457-i466.

4. Weinstein, J. N., et al. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. *Nature Genetics*, 45(10), 1113-1120.

5. Yang, W., et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery. *Nucleic Acids Research*, 41(D1), D955-D961.

---

**본 프로젝트는 암 환자의 생명을 구하고 삶의 질을 향상시키는 것을 목표로 합니다.**

**Last Updated: 2024-12-30**
