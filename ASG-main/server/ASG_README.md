## 📁 API Base URL

```
http://127.0.0.1:5000
```

---

## 1. 로그인

**POST /login**

사용자 로그인

### 요청 예시

```json
{
    "username": "20210030",
    "password": "miMI121832!"
}
```

### 응답 예시

```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTc2MTUzOTczNCwianRpIjoiYzI1OTY0MzQtN2FmZC00YjU1LThlYTMtMzI4MWE3MTFiMDkwIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IjIwMjEwMDMwIiwibmJmIjoxNzYxNTM5NzM0LCJjc3JmIjoiY2Y3ZjZjNzYtOTMyYy00ODk4LWJiNzctMzhmYmIzNzFiMDcwIiwiZXhwIjoxNzYxNTQxNTM0fQ.CXu3WHTInXdBtf_nWCx3yNEK1YjN1F4uKRqtOw1xMHE",
    "message": "로그인 성공",
    "user": {
        "id": 1,
        "username": "20210030",
        "video": "uploaded_videos\\20210030_10-1_Cam02_Assault03_Place07_Night_Summer.mp4"
    }
}
```

### 설명

| 필드명         | 타입   | 설명               |
| -------------- | ------ | ------------------ |
| `message`      | string | 로그인 성공 메시지 |
| `user`         | object | 사용자 정보        |
| `access_token` | string | JWT 액세스 토큰    |

## 2. 회원가입

**POST /register**

사용자 회원가입

### 요청 예시

```json
{
    "username": "soso",
    "password": "123456",
    "passwordCheck": "123456"
}
```

### 응답 예시

```json
{
    "message": "soso signed up successfully!",
    "user": {
        "id": 2,
        "username": "soso",
        "video": ""
    }
}
```

### 설명

| 필드명    | 타입   | 설명                 |
| --------- | ------ | -------------------- |
| `message` | string | 회원가입 성공 메시지 |
| `user`    | object | 사용자 정보          |

---

## 3. 영상 추가

**POST /uploadVideo**

인증된 사용자가 자신의 계정에 1개의 동영상만 업로드할 수 있는 API입니다.
이미 업로드된 동영상이 있을 경우, 새 업로드는 거부됩니다.

### 요청 예시

```json
json이 아니라서 그냥 '박소정'한테 물어보세요 친절하게 알려드림
```

### 응답 예시

```json
{
    "message": "Video uploaded successfully",
    "user": {
        "full_path": "C:\\Users\\sojeong\\Documents\\GitHub\\ASG\\server\\uploaded_videos\\soso_10-1_Cam02_Assault03_Place07_Night_Summer.mp4",
        "id": 2,
        "username": "soso",
        "video": "uploaded_videos\\soso_10-1_Cam02_Assault03_Place07_Night_Summer.mp4"
    }
}
```

### 설명

| 필드명    | 타입   | 설명                 |
| --------- | ------ | -------------------- |
| `message` | string | 회원가입 성공 메시지 |
| `user`    | object | 사용자 정보          |

---

## 3. 영상 삭제

**DELETE /bringVideo**

영상 삭제

### 요청 예시

```json
json이 아니라서 그냥 '박소정'한테 물어보세요 친절하게 알려드림
```

### 응답 예시

```json
{
    "message": "Video deleted successfully"
}
```

### 설명

| 필드명    | 타입   | 설명                 |
| --------- | ------ | -------------------- |
| `message` | string | 영상삭제 성공 메시지 |

---

## 3. 영상 가져오기

**GET /bringVideo?username=soso**

특정 사용자의 업로드된 동영상을 서버에서 가져옴

### 응답 예시

```json
{
    "hasVideo": false,
    "message": "해당 유저는 영상이 없습니다."
}
```

### 설명

| 필드명     | 타입   | 설명                 |
| ---------- | ------ | -------------------- |
| `hasVideo` | string | 영상 여부            |
| `message`  | string | 영상가져 성공 메시지 |

---
