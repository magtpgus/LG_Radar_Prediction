# LG Radar Prediction

📡 Radar 성능 예측 AI - LG AI 경진대회 (Team 해커톤)

본 프로젝트는 2022 LG AI 경진대회에서 Team 해커톤이 개발한 AI 모델로, 자율주행용 Radar 센서의 안테나 성능을 공정 데이터를 바탕으로 예측하는 머신러닝 파이프라인입니다. 
---

📌 프로젝트 개요

- 주제: Radar 센서 안테나의 14개 성능 지표를 다중 회귀 방식으로 예측
- 데이터:
  - `train.csv` - 학습 데이터 (총 39,607개 샘플, X Feature 56개, Y Feature 14개)
  - `test.csv` - 테스트 데이터 (총 39,608개 샘플, X Feature 56개)
  - `y_feature_spec_info.csv` - Y Feature별 정상/불량 기준값 정보
  - `sample_submission.csv` - 제출 양식
  - dacon 에서 다운로드 가능

---

💡 주요 전처리 및 모델링 과정

🔧 데이터 전처리
- 파생 변수 생성:
  - 특정 X Feature 간 비율, 합계, 차이 기반 도메인 특성 반영
  - 예: `X_03 / X_07`, `X_41~X_44` 구간 차이, `X_1~6` 누름량 합산 등
- Feature 선택:
  - 통계 기반 상관관계 분석 및 중요도 기반 Feature 제거

---

📈 모델 학습 및 평가
- 사용 모델:
  - CatBoostRegressor
  - LightGBM
  - XGBoost
- 앙상블 전략:
  - `StratifiedKFold` 기반 6-Fold 교차 검증
  - 각 Fold 결과 평균을 통한 Soft Voting 예측
- 평가지표:
  - NRMSE: 중요 Y Feature에 1.2 가중치를 부여한 평가 방식 사용

---

🧱 코드 구조
- config.py : 실험 파라미터 및 드롭할 Feature 설정 
- data_loader.py : 데이터 로딩 및 라벨 생성
- feature_engineer.py : 파생변수 생성 및 통계 처리
- model_wrappers.py : LGBM, XGB 모델 커스텀 래퍼 클래스 
- utils.py : NRMSE 계산 함수 
- train.py : 전체 학습 및 예측 파이프라인
- train.csv, test.csv, y_feature_spec_info.csv 
- sample_submission.csv : 제출 형식 파일 


🏆 참여 정보
- 대회명: LG AI 경진대회 - Radar 성능 예측
- 기간: 2022년 8월 1일 ~ 2022년 8월 26일
- 주최/주관: LG AI Research / Dacon
- 팀명: 해커톤
- 최종결과 : 2위
