# AAC Scene Recognition AI

학습을 완료한 CLIP 기반 장면 분류 모델을 FastAPI로 제공합니다. 백엔드는 이미지를 전송하고, API는 장면 이름과 예측 확률을 반환합니다.

## 분류 클래스

- Cafe
- Convenience Store
- Hospital
- Pharmacy
- Public Transport
- Restaurant

## 구성

```text
app/main.py            FastAPI 서버 및 추론 로직
models/clip.py         학습에 사용한 CLIPClassifier 정의
weights/best_model.pt  학습 완료 체크포인트
```

서버 시작 시 `weights/best_model.pt`를 한 번만 메모리에 로드합니다. CUDA를 사용할 수 있으면 GPU를, 아니면 CPU를 사용합니다.

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

서버가 실행되면 Swagger UI는 <http://127.0.0.1:8000/docs>에서 확인할 수 있습니다.

## API

### `GET /`

API 접속 여부를 확인합니다.

```json
{
  "message": "AAC Scene Recognition API"
}
```

### `GET /health`

서버 상태를 확인합니다.

```json
{
  "status": "ok"
}
```

### `POST /predict`

`multipart/form-data`의 `file` 필드로 이미지(JPEG, PNG, WebP, BMP)를 전송합니다.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample.jpg"
```

응답 예시: 

```json
{
  "scene": "Cafe",
  "score": 0.93
}
```

- `scene`: 가장 높은 확률의 분류 클래스
- `score`: 해당 클래스의 softmax 확률(소수점 넷째 자리까지)

유효하지 않은 파일은 `400`, 이미지가 아닌 업로드는 `415`를 반환합니다. 프런트엔드/백엔드의 다른 Origin 연동을 위해 CORS를 허용합니다.
