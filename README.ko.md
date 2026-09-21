# FedVar

[English](README.md) | **한국어**

**FedVar: Federated Learning Algorithm with Weight Variation in Clients**의 저자 공식 구현입니다. 클라이언트 모델의 가중치 노름이 평균 ± 표준편차 범위에 포함되는지 판단하고, 선택된 모델을 평균하여 비 IID 환경의 연합학습을 수행합니다.

**Wooseok Shin, Jitae Shin · ITC-CSCC 2022 · pp. 456–459**

[논문 DOI](https://doi.org/10.1109/ITC-CSCC55581.2022.9894899) · [저장소의 논문 PDF](FedVar__Federated_Learning_Algorithm_with_Weight_Variation_in_Clients.pdf) · [알고리즘 설명](docs/ALGORITHM.md) · [실험 안내](docs/EXPERIMENTS.md)

## 빠른 실행

Python 3.10 이상이 필요합니다. 아래 예제는 CPU에서 실행되며 데이터 다운로드가 필요 없습니다.

```bash
git clone https://github.com/wsshinskku/FedVar.git
cd FedVar
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
fedvar train --config configs/smoke.json --output runs/smoke
fedvar evaluate --checkpoint runs/smoke/checkpoint.pt
```

명령 실행 파일을 사용할 수 없는 환경에서는 `python -m fedvar`를 사용합니다. 기존 파일명도 `python FedVar.py train --config configs/smoke.json --output runs/legacy-entry` 형태로 사용할 수 있습니다.

## MNIST와 비교 실험

```bash
python -m pip install -e '.[vision]'
fedvar train --config configs/mnist.json --download --output runs/mnist
fedvar compare --config configs/mnist.json --download --output runs/comparison
```

`compare`는 FedAvg, FedSGD, FedProx, FedVar에 동일한 초기화·데이터 분할·참여 클라이언트 시드를 적용합니다. MNIST, CIFAR-10, CIFAR-100을 지원하며 `--dataset cifar10`처럼 바꿀 수 있습니다. `--download`를 명시해야 데이터 다운로드를 허용합니다. `--device cuda`로 GPU를 선택할 수 있습니다.

기본 CNN·MLP 외에 `pip install -e '.[mobile]'`로 TinyNet, GhostNet, MobileNetV3 계열을 사용할 수 있습니다. 구체적인 모델명과 배치 정규화 관련 설정은 [실험 안내](docs/EXPERIMENTS.md)에 정리했습니다.

## 알고리즘

참여 클라이언트의 전체 학습 파라미터를 펼친 벡터를 `w_k`, 그 L2 노름을 `z_k`라고 하면:

```text
평균 = mean(z_k)
표준편차 = sqrt(mean((z_k - 평균)^2))
선택 집합 = {k: 평균 - 표준편차 <= z_k <= 평균 + 표준편차}
다음 서버 모델 = 선택된 클라이언트 모델의 산술 평균
```

논문의 Algorithm 2에 따라 모집단 표준편차와 경계를 포함하는 조건을 사용합니다. 노름은 클라이언트를 선택하는 데 사용하고, 실제 모델 텐서를 평균합니다. FedVar는 선택된 클라이언트에 동일한 가중치를 적용하며, 비교용 FedAvg는 샘플 수에 비례한 가중치를 적용합니다. 매 라운드 클라이언트 모델은 서버 모델에서 독립적으로 복사하고, 집계 후 서버 모델 자체를 갱신하여 별도의 테스트 데이터에서 평가합니다.

## 설정과 결과

| 설정 | 용도 |
|---|---|
| `configs/smoke.json` | 합성 데이터로 전체 실행 흐름 확인 |
| `configs/mnist.json` | 20개 클라이언트, 20라운드의 MNIST 실험 |
| `configs/paper-budget-mnist.json` | 논문의 100개 클라이언트·10개 참여·200라운드·로컬 5에폭·배치 50·20라운드 평가 주기 |

마지막 설정은 논문의 학습 규모에 MNIST·CNN·Dirichlet 0.5 분할을 적용합니다. 각 실행은 모델 구조, 최적화 설정, 무작위 시드와 정확한 클라이언트 데이터 분할을 저장하므로 실험 조건을 확인하고 반복할 수 있습니다.

각 실행은 설정, 클라이언트별 인덱스·클래스 분포, 라운드별 정확도·손실·선택 노름, CSV, 최종 서버 체크포인트를 저장합니다. 기존 결과 디렉터리는 덮어쓰지 않습니다.

## 구조와 검증

```text
src/fedvar/        집계, 데이터 분할, 모델, 학습, CLI
configs/           실행 설정
tests/             알고리즘·학습·체크포인트 검증
docs/              수식 대응 및 실험 안내
legacy/            기존 FedVar.py 원본 보존
```

```bash
pytest -q
ruff check .
```

기존 코드의 노름 비율 가중 합산, 클라이언트 간 모델 공유, 첫 모델의 텐서 변경, 누락된 서버 집계 및 정의되지 않은 모델·생성자 인자를 수정했습니다. 원본 코드는 [legacy/FedVar_original.py](legacy/FedVar_original.py)에 그대로 보존되어 있습니다.

## 인용

```bibtex
@inproceedings{shin2022fedvar,
  author = {Shin, Wooseok and Shin, Jitae},
  title = {FedVar: Federated Learning Algorithm with Weight Variation in Clients},
  booktitle = {2022 37th International Technical Conference on Circuits/Systems,
               Computers and Communications (ITC-CSCC)},
  year = {2022},
  pages = {456--459},
  doi = {10.1109/ITC-CSCC55581.2022.9894899}
}
```

[CITATION.cff](CITATION.cff)에서도 동일한 인용 정보를 제공합니다.
