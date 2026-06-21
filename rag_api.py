import json
import os
import time
import hmac
import hashlib
import base64
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor

import requests
from chromadb import PersistentClient
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "fraud_cases")
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

# HuggingFace 镜像加速（国内推荐），不强制 OFFLINE 以便首次下载模型
# 首次运行后若需严格离线，可手动设置 HF_HUB_OFFLINE=1
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

load_dotenv()

# ─────────────────────────────────────────────
# DeepSeek（RAG 问答）
# ─────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
else:
    print("警告：未找到 DEEPSEEK_API_KEY，/chat 接口将不可用。")

# ─────────────────────────────────────────────
# Doubao 图片识别
# ─────────────────────────────────────────────
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
DOUBAO_BASE_URL = os.getenv(
    "DOUBAO_BASE_URL",
    "https://ark.cn-beijing.volces.com/api/v3/responses"
)

# ─────────────────────────────────────────────
# 讯飞语音：一套凭证，支持四种识别场景
# ─────────────────────────────────────────────
IFLYTEK_APP_ID = os.getenv("IFLYTEK_APP_ID", "")
IFLYTEK_API_KEY = os.getenv("IFLYTEK_API_KEY", "")
IFLYTEK_API_SECRET = os.getenv("IFLYTEK_API_SECRET", "")

IFLYTEK_LANGUAGE_MAP = {
    "mandarin":   "zh_cn",
    "cantonese":  "zh_cn",
    "lmz":        "zh_cn",
    "henanese":   "zh_cn",
    "northeast":  "zh_cn",
    "en":         "en",
}

IFLYTEK_ENGINE_MAP = {
    "mandarin":   "sms16k",
    "cantonese":  "sms16k",
    "lmz":        "sms16k",
    "henanese":   "sms16k",
    "northeast":  "sms16k",
    "en":         "sms16k",
}

IFLYTEK_MODES = {
    "iat":     {"name": "语音听写（实时）",     "default_accent": "mandarin"},
    "dialect": {"name": "方言识别（实时）",     "default_accent": "cantonese"},
    "cn":      {"name": "中文识别（实时）",     "default_accent": "mandarin"},
    "lfasr":   {"name": "录音文件转写（异步）", "default_accent": "mandarin"},
}

# ─────────────────────────────────────────────
# RAG 模型（Embedding + Reranker）
# 模型本地路径：models/ 目录，首次使用前通过 HuggingFace 镜像下载
# 也可直接写 repo ID，sentence-transformers 会自动缓存到 ~/.cache/
def _resolve_model_path(env_key: str, default: str) -> str:
    path = os.getenv(env_key, "")
    if not path:
        return default
    if Path(path).exists():
        return path
    if "/" in path:
        return path
    return default

EMBEDDING_MODEL_NAME = _resolve_model_path("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_NAME = _resolve_model_path("RERANKER_MODEL", "BAAI/bge-reranker-base")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "15"))          # Stage 1 粗召回数量
RAG_RERANK_TOP = int(os.getenv("RAG_RERANK_TOP", "5")) # Stage 2 精排后返回数量
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 精排分数阈值（0~1）

# 延迟加载
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"加载 Embedding 模型：{EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        print(f"加载 Reranker 模型：{RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


# ─────────────────────────────────────────────
# 微信小程序：下载临时文件转 base64
# ─────────────────────────────────────────────
WECHAT_APPID = os.getenv("WECHAT_APPID", "")
WECHAT_SECRET = os.getenv("WECHAT_SECRET", "")
_wx_access_token_cache = {"token": "", "expire_at": 0}


def get_wx_access_token():
    """获取并缓存小程序 access_token"""
    now = time.time()
    if _wx_access_token_cache["token"] and _wx_access_token_cache["expire_at"] > now + 60:
        return _wx_access_token_cache["token"]
    if not WECHAT_APPID or not WECHAT_SECRET:
        raise RuntimeError("未配置 WECHAT_APPID / WECHAT_SECRET")
    url = (
        f"https://api.weixin.qq.com/cgi-bin/token"
        f"?grant_type=client_credential&appid={WECHAT_APPID}&secret={WECHAT_SECRET}"
    )
    resp = requests.get(url, timeout=10).json()
    token = resp.get("access_token")
    expires_in = resp.get("expires_in", 7200)
    if not token:
        raise RuntimeError(f"获取 access_token 失败: {resp}")
    _wx_access_token_cache["token"] = token
    _wx_access_token_cache["expire_at"] = now + expires_in
    return token


def download_wxmp_file(file_id: str) -> str:
    """下载小程序 wx.uploadFile 的临时文件，返回 base64 字符串"""
    token = get_wx_access_token()
    url = f"https://api.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={file_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"下载小程序文件失败: HTTP {resp.status_code}")
    return base64.b64encode(resp.content).decode("utf-8")


# ─────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────
CHROMA_CLIENT = PersistentClient(path=str(CHROMA_DIR))
COLLECTION = CHROMA_CLIENT.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

LOW_RISK_INPUT_KEYWORDS = [
    "淘宝", "京东", "红包", "返现", "提现", "优惠券", "积分", "抽奖",
    "问卷", "调研", "拒收回复R", "掌银", "手机银行", "理财产品",
    "全球通", "唯品会", "得物", "支付宝红包", "信用卡", "绑定微信",
    "百万保障", "安全认证", "人脸识别", "更新信息", "完善资料", "账号异常"
]
HIGH_RISK_KEYWORDS = [
    "转账", "验证码", "下载", "app", "屏幕共享", "安全账户",
    "贷款", "刷单", "冒充", "冻结", "解冻", "保证金", "安全账户",
    "远程控制", "共享桌面", "呼叫转移", "安全核查", "资金核查"
]


def get_case_count() -> int:
    try:
        return COLLECTION.count()
    except Exception:
        return 0


def is_low_risk_by_rules(query, best_match):
    """低风险兜底规则：查询和案例均无高危信号时才通过"""
    if best_match and best_match.get("score", 0) >= 0.3:
        case_type = best_match.get("type", "")
        if any(kw in case_type for kw in ["正常", "营销", "服务通知", "金融营销", "低风险"]):
            if not any(kw in query for kw in HIGH_RISK_KEYWORDS):
                matched = [kw for kw in LOW_RISK_INPUT_KEYWORDS if kw in query]
                if matched:
                    reason = (
                        f"您输入的内容包含\"{', '.join(matched[:3])}\"等正常业务特征，"
                        "未发现高危关键词，故判定为低风险。请通过官方渠道核实活动真实性。"
                    )
                    return True, reason
    if not best_match:
        matched = [kw for kw in LOW_RISK_INPUT_KEYWORDS if kw in query]
        if matched and not any(kw in query for kw in HIGH_RISK_KEYWORDS):
            reason = (
                f"未检索到相似诈骗案例，您输入的内容包含\"{', '.join(matched[:3])}\"等正常业务特征，"
                "请通过官方 App 或客服电话核实。"
            )
            return True, reason
    return False, ""


# ─────────────────────────────────────────────
# RAG：两阶段检索（Embedding 召回 → Reranker 精排）
# ─────────────────────────────────────────────
def build_results(query, top_k=RAG_TOP_K, rerank_top=RAG_RERANK_TOP,
                  score_threshold=RAG_SCORE_THRESHOLD):
    """
    两阶段检索流程：
    1. Embedding 向量检索，召回 top_k 条
    2. Reranker 对召回结果精排，返回 rerank_top 条
    """
    model = get_embedding_model()
    reranker = get_reranker_model()

    # Stage 1：向量检索
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    response = COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, RAG_TOP_K),
        include=["documents", "metadatas", "distances"],
    )
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    if not documents:
        return []

    # Stage 2：精排（pairwise 打分）
    pairs = [[query, doc] for doc in documents]
    rerank_scores = reranker.predict(pairs)

    # 组装结果，按精排分数降序
    scored_results = []
    for idx, doc in enumerate(documents):
        metadata = metadatas[idx] or {}
        distance = float(distances[idx]) if idx < len(distances) else 1.0
        embed_score = max(0.0, 1.0 - distance)
        rerank_score = float(rerank_scores[idx])
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        scored_results.append({
            "score": rerank_score,
            "embed_score": embed_score,
            "type": metadata.get("type", "未分类"),
            "stage": metadata.get("stage", ""),
            "source": metadata.get("source", ""),
            "keywords": keywords,
            "text": (metadata.get("full_text") or doc or "")[:800],
            "_document": doc,
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = []
    for i, item in enumerate(scored_results):
        if len(final_results) >= rerank_top:
            break
        if item["score"] >= score_threshold:
            item["rank"] = i + 1
            final_results.append(item)

    return final_results


def load_system_prompt():
    p = BASE_DIR / "prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "你是反诈助手。"


def call_deepseek(query, results):
    if not deepseek_client:
        raise RuntimeError("DeepSeek API Key 未配置，请在 .env 文件中设置 DEEPSEEK_API_KEY")
    system_prompt = load_system_prompt()
    if results:
        cases_text = "".join(
            f"\n案例{idx + 1}（{c['type']}，精排分数 {c['score']:.3f}）：\n{c['text']}\n"
            for idx, c in enumerate(results[:3])
        )
    else:
        cases_text = "未检索到高度相似的案例。"
    user_content = (
        f"用户问题：{query}\n\n"
        f"本地检索到的相似案例：{cases_text}\n\n"
        "请分析风险等级（high/medium/low）、诈骗类型、理由列表、提醒列表，输出 JSON。"
    )
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────────
# 讯飞：生成 WebSocket 鉴权 URL
# ─────────────────────────────────────────────
def build_iflytek_ws_url(mode: str = "iat", accent: str = "mandarin"):
    if not IFLYTEK_APP_ID or not IFLYTEK_API_KEY or not IFLYTEK_API_SECRET:
        raise RuntimeError(
            "讯飞凭证未完整配置，请在 .env 中设置 "
            "IFLYTEK_APP_ID、IFLYTEK_API_KEY、IFLYTEK_API_SECRET。"
        )
    cfg = IFLYTEK_MODES.get(mode, IFLYTEK_MODES["iat"])

    language = IFLYTEK_LANGUAGE_MAP.get(accent, "zh_cn")
    engine = IFLYTEK_ENGINE_MAP.get(accent, "sms16k")

    ts = str(int(time.time()))
    sign_str = IFLYTEK_APP_ID + ts
    sign = base64.b64encode(
        hmac.new(IFLYTEK_API_SECRET.encode("utf-8"), sign_str.encode("utf-8"),
                 digestmod=hashlib.sha256).digest()
    ).decode("utf-8")

    params = f"appid={IFLYTEK_APP_ID}&ts={ts}&signa={requests.utils.quote(sign)}"
    ws_url = f"wss://iat-api.xfyun.cn/v2/iat?{params}"

    return ws_url, cfg["name"], accent, language, engine


# ─────────────────────────────────────────────
# 讯飞：录音文件异步转写（LFASR）
# ─────────────────────────────────────────────
def call_iflytek_lfasr(file_bytes: bytes, file_name: str):
    if not IFLYTEK_APP_ID or not IFLYTEK_API_SECRET:
        raise RuntimeError(
            "讯飞凭证未配置，请在 .env 中设置 "
            "IFLYTEK_APP_ID 和 IFLYTEK_API_SECRET。"
        )

    import tempfile, subprocess

    def convert_to_wav(audio_data: bytes, suffix: str = ".webm") -> bytes:
        tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_in.write(audio_data)
        tmp_in.close()
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out.close()
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in.name,
                 "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                 tmp_out.name],
                capture_output=True, timeout=60
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg 转换失败：{result.stderr.decode('utf-8', errors='ignore')}")
            with open(tmp_out.name, "rb") as f:
                return f.read()
        finally:
            Path(tmp_in.name).unlink(missing_ok=True)
            Path(tmp_out.name).unlink(missing_ok=True)

    wav_bytes = convert_to_wav(file_bytes, Path(file_name).suffix or ".webm")

    prepare_url = "https://raasr.xfyun.cn/api/prepare"
    ts = str(int(time.time()))
    sign_str = f"{IFLYTEK_APP_ID}{ts}"
    sign = base64.b64encode(
        hmac.new(IFLYTEK_API_SECRET.encode("utf-8"), sign_str.encode("utf-8"),
                 digestmod=hashlib.sha256).digest()
    ).decode("utf-8")

    wav_name = "audio.wav"

    try:
        import wave as _wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        with _wave.open(tmp_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = int(frames / float(rate))
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        duration = int(len(wav_bytes) / 32000)

    prep_payload = {
        "appId": IFLYTEK_APP_ID, "signa": sign, "ts": ts,
        "fileSize": len(wav_bytes), "fileName": wav_name,
        "duration": str(duration)
    }
    prep_resp = requests.post(prepare_url, data=prep_payload, timeout=30).json()
    if prep_resp.get("code") != "0":
        raise RuntimeError(f"讯飞转写准备失败：{prep_resp}")
    order_id = prep_resp["data"]

    up_resp = requests.post(
        "https://raasr.xfyun.cn/api/upload",
        files={"file": (wav_name, wav_bytes, "audio/wav")},
        data={"appId": IFLYTEK_APP_ID, "signa": sign, "ts": ts, "orderId": order_id},
        timeout=60
    ).json()
    if up_resp.get("code") != "0":
        raise RuntimeError(f"讯飞音频上传失败：{up_resp}")

    for _ in range(60):
        time.sleep(5)
        q_data = requests.post(
            "https://raasr.xfyun.cn/api/getProgress",
            data={"appId": IFLYTEK_APP_ID, "signa": sign, "ts": ts, "orderId": order_id},
            timeout=15
        ).json()
        status = q_data.get("data", {}).get("status", 0)
        if status == 9:
            break
        elif status not in (1, 2):
            raise RuntimeError(f"讯飞转写异常，状态码：{status}")

    r_data = requests.post(
        "https://raasr.xfyun.cn/api/getResult",
        data={"appId": IFLYTEK_APP_ID, "signa": sign, "ts": ts, "orderId": order_id},
        timeout=30
    ).json()
    if r_data.get("code") != "0":
        raise RuntimeError(f"讯飞获取结果失败：{r_data}")

    text_parts = []
    for seg in r_data.get("data", []):
        for w in seg.get("wb", []):
            text_parts.append(w.get("w", ""))
    return "".join(text_parts)


# ─────────────────────────────────────────────
# Doubao 图片识别
# ─────────────────────────────────────────────
def call_doubao_vision(base64_image: str, prompt: str = None):
    if not DOUBAO_API_KEY:
        raise RuntimeError("Doubao API Key 未配置，请在 .env 中设置 DOUBAO_API_KEY")
    if prompt is None:
        prompt = (
            "请分析这张截图或图片中的文字内容，"
            "识别其中是否存在诈骗风险，例如：冒充客服、钓鱼链接、恐吓威胁、诱导转账等。"
            "如果发现可疑内容请详细说明。"
        )
    payload = {
        "model": DOUBAO_MODEL,
        "input": {
            "prompt": prompt,
            "images": [{"type": "base64", "data": base64_image}]
        },
        "parameters": {"do_sample": False},
        "stream": False
    }
    headers = {"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(DOUBAO_BASE_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Doubao 请求失败：{resp.status_code} {resp.text}")
    result = resp.json()
    outputs = result.get("output", {}).get("choices", [{}])
    if not outputs:
        raise RuntimeError(f"Doubao 返回格式异常：{result}")
    return outputs[0].get("message", {}).get("content", "（未识别到内容）")


# ─────────────────────────────────────────────
# HTTP 服务
# ─────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/health":
            self._send_json(200, {
                "ok": True,
                "cases": get_case_count(),
                "collection": COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "reranker_model": RERANKER_MODEL_NAME,
            })
            return

        if path == "/debug":
            q = qs.get("q", [None])[0]
            if not q:
                self._send_json(400, {"detail": "请提供 ?q=查询内容"})
                return
            raw = _debug_search(q)
            self._send_json(200, raw)
            return

        if path == "/iflytek/modes":
            shared_configured = bool(IFLYTEK_APP_ID and IFLYTEK_API_KEY and IFLYTEK_API_SECRET)
            modes_info = {}
            for mode, cfg in IFLYTEK_MODES.items():
                modes_info[mode] = {
                    "name": cfg["name"],
                    "configured": shared_configured,
                    "default_accent": cfg["default_accent"],
                }
            self._send_json(200, {"modes": modes_info})
            return

        if path == "/iflytek/ws-url":
            mode = qs.get("mode", ["iat"])[0]
            accent = qs.get("accent", ["mandarin"])[0]
            try:
                url, name, accent, language, engine = build_iflytek_ws_url(mode, accent)
                self._send_json(200, {
                    "ws_url": url,
                    "mode": mode,
                    "name": name,
                    "accent": accent,
                    "language": language,
                    "engine": engine,
                })
            except RuntimeError as e:
                self._send_json(400, {"detail": str(e)})
            return

        self._send_json(404, {"detail": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ── 图片识别 ──────────────────────────────
        if path == "/image":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            base64_image = data.get("image") or data.get("image_base64") or ""
            if not base64_image:
                self._send_json(400, {"detail": "image 字段不能为空"})
                return
            try:
                result = call_doubao_vision(base64_image, data.get("prompt"))
                self._send_json(200, {"result": result})
            except RuntimeError as e:
                self._send_json(502, {"detail": str(e)})
            return

        # ── 讯飞录音文件转写 ───────────────────────
        if path == "/lfasr":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            audio_b64 = data.get("audio") or data.get("audio_base64") or ""
            file_name = data.get("filename", "audio.webm")
            if not audio_b64:
                self._send_json(400, {"detail": "audio 字段不能为空"})
                return
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                self._send_json(400, {"detail": "音频 Base64 解码失败"})
                return
            try:
                text = call_iflytek_lfasr(audio_bytes, file_name)
                self._send_json(200, {"text": text})
            except RuntimeError as e:
                self._send_json(502, {"detail": str(e)})
            return

        # ── RAG 文字问答 ──────────────────────────
        if path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            query = data.get("query", "").strip()
            if not query:
                self._send_json(400, {"detail": "query cannot be empty"})
                return
            try:
                results = build_results(query)
            except Exception as exc:
                self._send_json(500, {"detail": f"检索失败: {exc}"})
                return
            best_match = results[0] if results else None
            is_low, low_reason = is_low_risk_by_rules(query, best_match)
            if is_low:
                answer = {
                    "risk_level": "low",
                    "fraud_type": "正常营销/服务通知",
                    "reasons": [
                        low_reason,
                        "请通过官方 App 或客服电话核实活动真实性，不要轻易点击短信中的链接。",
                    ],
                    "attention": [
                        "切勿向任何人透露验证码、密码",
                        "建议关闭短信中的陌生链接，通过官方渠道操作",
                    ],
                }
                self._send_json(200, {"answer": answer, "rag_results": results})
                return
            try:
                answer = call_deepseek(query, results)
                self._send_json(200, {"answer": answer, "rag_results": results})
            except Exception as exc:
                self._send_json(500, {"detail": f"DeepSeek 调用失败: {exc}"})
            return

        # ── 小程序专用：图片上传识别 ────────────────
        if path == "/mp/image":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            # 兼容小程序 fileID 或 base64
            base64_image = data.get("image_base64", "")
            if data.get("file_id"):
                try:
                    base64_image = download_wxmp_file(data["file_id"])
                except Exception as e:
                    self._send_json(502, {"detail": f"下载文件失败: {e}"})
                    return
            if not base64_image:
                self._send_json(400, {"detail": "image_base64 不能为空"})
                return
            try:
                result = call_doubao_vision(base64_image, data.get("prompt"))
                self._send_json(200, {"result": result})
            except RuntimeError as e:
                self._send_json(502, {"detail": str(e)})
            return

        # ── 小程序专用：语音文件转写 ────────────────
        if path == "/mp/lfasr":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"detail": "Invalid JSON"})
                return
            audio_b64 = data.get("audio_base64", "")
            file_name = data.get("filename", "audio.mp3")
            if not audio_b64:
                self._send_json(400, {"detail": "audio_base64 不能为空"})
                return
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                self._send_json(400, {"detail": "音频 Base64 解码失败"})
                return
            try:
                text = call_iflytek_lfasr(audio_bytes, file_name)
                self._send_json(200, {"text": text})
            except RuntimeError as e:
                self._send_json(502, {"detail": str(e)})
            return

        self._send_json(404, {"detail": "Not Found"})

    def log_message(self, fmt, *args):
        print(fmt % args)


def _debug_search(query_text: str):
    model = get_embedding_model()
    reranker = get_reranker_model()
    emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    raw = COLLECTION.query(
        query_embeddings=[emb],
        n_results=RAG_TOP_K,
        include=["documents", "distances"],
    )
    distances = raw.get("distances", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    if not docs:
        return {"query": query_text, "results": []}

    pairs = [[query_text, doc] for doc in docs]
    scores = reranker.predict(pairs)

    results = []
    for i, (doc, dist, score) in enumerate(zip(docs, distances, scores)):
        d = float(dist)
        results.append({
            "rank": i + 1,
            "rerank_score": round(float(score), 6),
            "embed_score": round(max(0.0, 1.0 - d), 6),
            "doc_preview": (doc or "")[:120],
        })
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    for i, r in enumerate(results):
        r["rank_after_rerank"] = i + 1
    return {"query": query_text, "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME, "results": results[:10]}


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"RAG 服务运行在 http://{HOST}:{PORT}，集合：{COLLECTION_NAME}")
    print(f"Embedding 模型：{EMBEDDING_MODEL_NAME}")
    print(f"Reranker 模型：{RERANKER_MODEL_NAME}")

    # 预加载模型（避免第一次请求时冷启动卡住）
    print("正在后台预加载模型，请稍候...")
    def preload():
        try:
            get_embedding_model()
            print("Embedding 模型加载完成")
        except Exception as e:
            print(f"Embedding 模型加载失败：{e}")
        try:
            get_reranker_model()
            print("Reranker 模型加载完成")
        except Exception as e:
            print(f"Reranker 模型加载失败：{e}")
    threading.Thread(target=preload, daemon=True).start()

    print("  /chat              - RAG 文字问答（两阶段：召回→精排）")
    print("  /image             - Doubao 图片识别（POST: {image: base64}）")
    print("  /mp/image          - 小程序专用图片识别（POST: {image_base64 或 file_id}）")
    print("  /mp/lfasr          - 小程序专用语音转写（POST: {audio_base64, filename}）")
    print("  /lfasr             - 讯飞录音文件转写（POST: {audio: base64, filename}）")
    print("  /iflytek/modes     - 查询讯飞凭证状态")
    print("  /iflytek/ws-url   - 获取讯飞实时听写 WebSocket URL（?mode=iat&accent=mandarin）")
    print("  /debug?q=...       - 诊断检索效果（显示精排分数）")
    print("  /health            - 健康检查")
    server.serve_forever()


# ─────────────────────────────────────────────
# WSGI 入口（给 waitress / 微信云托管用）
# ─────────────────────────────────────────────
def _parse_query(qs: str) -> dict:
    """把 query string 转成 dict（支持重复 key）"""
    from urllib.parse import parse_qs
    parsed = parse_qs(qs, keep_blank_values=True)
    return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}


def _read_json_body(body: bytes) -> dict:
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {}


def _json_response(status: int, payload: dict):
    """生成 (status, headers, body) 三元组给 WSGI start_response"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]
    return status, headers, body


def _dispatch(method: str, path: str, qs: str, body: bytes) -> tuple:
    """
    直接实现路由逻辑（不依赖 BaseHTTPRequestHandler）。
    返回 (status, headers, body_bytes)。
    """
    query = _parse_query(qs)

    # /health
    if method == "GET" and path == "/health":
        return _json_response(200, {
            "ok": True,
            "cases": get_case_count(),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
        })

    # /debug
    if method == "GET" and path == "/debug":
        q = query.get("q")
        if not q:
            return _json_response(400, {"detail": "请提供 ?q=查询内容"})
        try:
            return _json_response(200, _debug_search(q))
        except Exception as e:
            return _json_response(500, {"detail": f"debug 失败: {e}"})

    # /iflytek/modes
    if method == "GET" and path == "/iflytek/modes":
        shared_configured = bool(IFLYTEK_APP_ID and IFLYTEK_API_KEY and IFLYTEK_API_SECRET)
        modes_info = {}
        for mode, cfg in IFLYTEK_MODES.items():
            modes_info[mode] = {
                "name": cfg["name"],
                "configured": shared_configured,
                "default_accent": cfg["default_accent"],
            }
        return _json_response(200, {"modes": modes_info})

    # /iflytek/ws-url
    if method == "GET" and path == "/iflytek/ws-url":
        mode = query.get("mode", "iat")
        accent = query.get("accent", "mandarin")
        try:
            url, name, accent, language, engine = build_iflytek_ws_url(mode, accent)
            return _json_response(200, {
                "ws_url": url, "mode": mode, "name": name,
                "accent": accent, "language": language, "engine": engine,
            })
        except RuntimeError as e:
            return _json_response(400, {"detail": str(e)})

    # /image
    if method == "POST" and path == "/image":
        data = _read_json_body(body)
        base64_image = data.get("image") or data.get("image_base64") or ""
        if not base64_image:
            return _json_response(400, {"detail": "image 字段不能为空"})
        try:
            result = call_doubao_vision(base64_image, data.get("prompt"))
            return _json_response(200, {"result": result})
        except RuntimeError as e:
            return _json_response(502, {"detail": str(e)})

    # /lfasr
    if method == "POST" and path == "/lfasr":
        data = _read_json_body(body)
        audio_b64 = data.get("audio") or data.get("audio_base64") or ""
        file_name = data.get("filename", "audio.webm")
        if not audio_b64:
            return _json_response(400, {"detail": "audio 字段不能为空"})
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return _json_response(400, {"detail": "音频 Base64 解码失败"})
        try:
            text = call_iflytek_lfasr(audio_bytes, file_name)
            return _json_response(200, {"text": text})
        except RuntimeError as e:
            return _json_response(502, {"detail": str(e)})

    # /mp/image
    if method == "POST" and path == "/mp/image":
        data = _read_json_body(body)
        base64_image = data.get("image_base64", "")
        if data.get("file_id"):
            try:
                base64_image = download_wxmp_file(data["file_id"])
            except Exception as e:
                return _json_response(502, {"detail": f"下载文件失败: {e}"})
        if not base64_image:
            return _json_response(400, {"detail": "image_base64 不能为空"})
        try:
            result = call_doubao_vision(base64_image, data.get("prompt"))
            return _json_response(200, {"result": result})
        except RuntimeError as e:
            return _json_response(502, {"detail": str(e)})

    # /mp/lfasr
    if method == "POST" and path == "/mp/lfasr":
        data = _read_json_body(body)
        audio_b64 = data.get("audio_base64", "")
        file_name = data.get("filename", "audio.mp3")
        if not audio_b64:
            return _json_response(400, {"detail": "audio_base64 不能为空"})
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            return _json_response(400, {"detail": "音频 Base64 解码失败"})
        try:
            text = call_iflytek_lfasr(audio_bytes, file_name)
            return _json_response(200, {"text": text})
        except RuntimeError as e:
            return _json_response(502, {"detail": str(e)})

    # /chat
    if method == "POST" and path == "/chat":
        data = _read_json_body(body)
        query_str = (data.get("query") or "").strip()
        if not query_str:
            return _json_response(400, {"detail": "query cannot be empty"})
        try:
            results = build_results(query_str)
        except Exception as exc:
            return _json_response(500, {"detail": f"检索失败: {exc}"})
        best_match = results[0] if results else None
        is_low, low_reason = is_low_risk_by_rules(query_str, best_match)
        if is_low:
            answer = {
                "risk_level": "low",
                "fraud_type": "正常营销/服务通知",
                "reasons": [
                    low_reason,
                    "请通过官方 App 或客服电话核实活动真实性，不要轻易点击短信中的链接。",
                ],
                "attention": [
                    "切勿向任何人透露验证码、密码",
                    "建议关闭短信中的陌生链接，通过官方渠道操作",
                ],
            }
            return _json_response(200, {"answer": answer, "rag_results": results})
        try:
            answer = call_deepseek(query_str, results)
            return _json_response(200, {"answer": answer, "rag_results": results})
        except Exception as exc:
            return _json_response(500, {"detail": f"DeepSeek 调用失败: {exc}"})

    return _json_response(404, {"detail": "Not Found"})


def wsgi_app(environ, start_response):
    """WSGI 入口：waitress / 微信云托管使用。"""
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    qs = environ.get("QUERY_STRING", "")

    if method == "OPTIONS":
        status, headers, body = _json_response(200, {"ok": True})
    else:
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
        except (TypeError, ValueError):
            content_length = 0
        body_bytes = b""
        if content_length > 0:
            try:
                body_bytes = environ["wsgi.input"].read(content_length)
            except Exception:
                body_bytes = b""

        try:
            status, headers, body = _dispatch(method, path, qs, body_bytes)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                with open(BASE_DIR / "wsgi_error.log", "a", encoding="utf-8") as f:
                    f.write(tb + "\n")
            except Exception:
                pass
            status, headers, body = _json_response(500, {"detail": f"server error: {e}"})

    status_text = {200: "OK", 400: "Bad Request", 404: "Not Found",
                   405: "Method Not Allowed", 500: "Internal Server Error",
                   502: "Bad Gateway"}.get(status, "OK")
    start_response(f"{status} {status_text}", headers)
    return [body]


# 兼容 waitress 入口（app = wsgi_app）
app = wsgi_app
