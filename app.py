import io
import re
import sys
import math
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="03B FontCLIP Brand Personality Analyzer", layout="wide")

APP_VERSION = "1.1"
APP_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path.home() / ".cache" / "fontclip_brand_personality_03b"

def resolve_resource(filename: str) -> Path:
    """Find a bundled resource even when Streamlit deploys app.py from a subfolder."""
    candidates = [
        APP_DIR / filename,
        Path.cwd() / filename,
        APP_DIR.parent / filename,
    ]
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        if p.exists() and p.is_file():
            return p

    # Last resort: search only inside the checked-out repository area.
    try:
        for p in APP_DIR.parent.rglob(filename):
            if p.is_file():
                return p
    except Exception:
        pass

    # Return the expected local path so later messages can show where it was expected.
    return APP_DIR / filename

# Same source/checkpoint as the completed 03A pilot.
FONTCLIP_COMMIT = "3d4c6af01f668800d8e4f9f4f753d29c74dad252"
FONTCLIP_ARCHIVE_URL = f"https://github.com/yukistavailable/FontCLIP/archive/{FONTCLIP_COMMIT}.zip"
FONTCLIP_CHECKPOINT_GDRIVE_ID = "1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1"
EXPECTED_CHECKPOINT_SHA256 = "c441277fbed4366d32d8fb65725189b97d3fe88bae5fe0648b969feea01bbb00"

MAPPING_FILE = resolve_resource("Aaker1997_42traits_15facets_5dimensions_mapping.csv")
DEFAULT_REVIEW_FILE = resolve_resource("03A_eligibility_review_final.csv")
DEFAULT_TEXT_PROFILE_FILE = resolve_resource("02_text_5D_raw_FOR_03B.csv")

DIMENSIONS = ["Sincerity", "Excitement", "Competence", "Sophistication", "Ruggedness"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# FontCLIP's own code uses attribute prompts such as "bold font".
# For the primary analysis, keep one fixed prompt family to minimize researcher degrees of freedom.
PROMPT_TEMPLATE = "{trait} font"
PROMPT_METHOD = "FontCLIP-native single positive attribute prompt: {trait} font"


EMBEDDED_AAKER_MAPPING_CSV = """Item_No,Dimension,Facet,Trait,Original_Facet_Code
1,Sincerity,Down-to-earth,down-to-earth,1a
2,Sincerity,Down-to-earth,family-oriented,1a
3,Sincerity,Down-to-earth,small-town,1a
4,Sincerity,Honest,honest,1b
5,Sincerity,Honest,sincere,1b
6,Sincerity,Honest,real,1b
7,Sincerity,Wholesome,wholesome,1c
8,Sincerity,Wholesome,original,1c
9,Sincerity,Cheerful,cheerful,1d
10,Sincerity,Cheerful,sentimental,1d
11,Sincerity,Cheerful,friendly,1d
12,Excitement,Daring,daring,2a
13,Excitement,Daring,trendy,2a
14,Excitement,Daring,exciting,2a
15,Excitement,Spirited,spirited,2b
16,Excitement,Spirited,cool,2b
17,Excitement,Spirited,young,2b
18,Excitement,Imaginative,imaginative,2c
19,Excitement,Imaginative,unique,2c
20,Excitement,Up-to-date,up-to-date,2d
21,Excitement,Up-to-date,independent,2d
22,Excitement,Up-to-date,contemporary,2d
23,Competence,Reliable,reliable,3a
24,Competence,Reliable,hard working,3a
25,Competence,Reliable,secure,3a
26,Competence,Intelligent,intelligent,3b
27,Competence,Intelligent,technical,3b
28,Competence,Intelligent,corporate,3b
29,Competence,Successful,successful,3c
30,Competence,Successful,leader,3c
31,Competence,Successful,confident,3c
32,Sophistication,Upper class,upper class,4a
33,Sophistication,Upper class,glamorous,4a
34,Sophistication,Upper class,good looking,4a
35,Sophistication,Charming,charming,4b
36,Sophistication,Charming,feminine,4b
37,Sophistication,Charming,smooth,4b
38,Ruggedness,Outdoorsy,outdoorsy,5a
39,Ruggedness,Outdoorsy,masculine,5a
40,Ruggedness,Outdoorsy,Western,5a
41,Ruggedness,Tough,tough,5b
42,Ruggedness,Tough,rugged,5b
"""

def load_mapping() -> pd.DataFrame:
    # Prefer the bundled CSV when present, but never crash at startup solely
    # because GitHub/Streamlit omitted a sidecar resource file.
    if MAPPING_FILE.exists():
        df = pd.read_csv(MAPPING_FILE)
    else:
        df = pd.read_csv(io.StringIO(EMBEDDED_AAKER_MAPPING_CSV))

    need = {"Item_No", "Dimension", "Facet", "Trait"}
    if not need.issubset(df.columns):
        raise ValueError(f"Aaker mapping is missing columns: {sorted(need - set(df.columns))}")
    df = df.sort_values("Item_No").reset_index(drop=True)
    return df


AAKER = load_mapping()
TRAITS = AAKER["Trait"].astype(str).tolist()
PROMPTS = [PROMPT_TEMPLATE.format(trait=t.lower()) for t in TRAITS]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def brand_key(s: str) -> str:
    s = str(s).lower().replace("ö", "o")
    return re.sub(r"[^a-z0-9]", "", s)


def basename_key(s: str) -> str:
    return Path(str(s)).name.lower()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_extract_zip(uploaded_bytes: bytes):
    out, skipped = [], []
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/"):
                continue
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            if info.file_size > 20 * 1024 * 1024:
                skipped.append({"Filename": name, "Reason": "File > 20 MB"})
                continue
            data = zf.read(info)
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception as e:
                skipped.append({"Filename": name, "Reason": f"Image decode failed: {e}"})
                continue
            out.append({"filename": name, "bytes": data, "image": img})
    return out, skipped


def infer_brand_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"(?i)_?logotype$", "", stem)
    return re.sub(r"[_-]+", " ", stem).strip()


def inspect_logo(item: dict) -> dict:
    img = item["image"]
    gray = img.convert("L")
    arr = np.asarray(gray)
    mask = arr < 245
    if mask.any():
        ys, xs = np.where(mask)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_max = max(bbox_w, bbox_h)
    else:
        bbox_w = bbox_h = bbox_max = 0
    return {
        "Filename": item["filename"],
        "Brand": infer_brand_from_filename(item["filename"]),
        "Neutral_Text": infer_brand_from_filename(item["filename"]),
        "Width": img.size[0],
        "Height": img.size[1],
        "Mode": img.mode,
        "Format": str(getattr(img, "format", "")),
        "Foreground_BBox_W": bbox_w,
        "Foreground_BBox_H": bbox_h,
        "Foreground_Max": bbox_max,
        "Standard_OK": bool(img.size == (1024, 1024) and 760 <= bbox_max <= 840),
        "Eligibility": "Unreviewed",
        "Researcher_Note": "",
    }


def merge_review(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    prev = previous.copy()
    prev["_base"] = prev["Filename"].map(basename_key) if "Filename" in prev.columns else ""
    by_base = prev.drop_duplicates("_base", keep="last").set_index("_base")
    for i, r in out.iterrows():
        k = basename_key(r["Filename"])
        if k not in by_base.index:
            continue
        pr = by_base.loc[k]
        for c in ["Brand", "Eligibility", "Researcher_Note"]:
            if c in prev.columns and pd.notna(pr.get(c)) and str(pr.get(c)).strip():
                out.at[i, c] = pr.get(c)
        out.at[i, "Neutral_Text"] = out.at[i, "Brand"]
    return out


def download_file(url: str, path: Path, timeout: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def prepare_fontclip_source() -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = RUNTIME_ROOT / "source" / f"FontCLIP-{FONTCLIP_COMMIT}"
    if source_root.exists() and (source_root / "models").exists():
        return source_root
    archive_path = RUNTIME_ROOT / f"fontclip-{FONTCLIP_COMMIT}.zip"
    if not archive_path.exists():
        download_file(FONTCLIP_ARCHIVE_URL, archive_path)
    parent = RUNTIME_ROOT / "source"
    parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(parent)
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("FontCLIP-") and (p / "models").exists()]
    if not candidates:
        raise RuntimeError("FontCLIP source archive structure could not be recognized.")
    return candidates[0]


def prepare_checkpoint() -> Path:
    ckpt = RUNTIME_ROOT / "model_checkpoints" / "model.pt"
    if ckpt.exists() and ckpt.stat().st_size > 10 * 1024 * 1024:
        if sha256_file(ckpt) == EXPECTED_CHECKPOINT_SHA256:
            return ckpt
        ckpt.unlink(missing_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    import gdown
    url = f"https://drive.google.com/uc?id={FONTCLIP_CHECKPOINT_GDRIVE_ID}"
    out = gdown.download(url, str(ckpt), quiet=False)
    if not out or not ckpt.exists() or ckpt.stat().st_size <= 10 * 1024 * 1024:
        raise RuntimeError("Official FontCLIP checkpoint download failed.")
    actual = sha256_file(ckpt)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch. Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual}.")
    return ckpt


@st.cache_resource(show_spinner=False)
def load_fontclip_runtime():
    repo_dir = prepare_fontclip_source()
    checkpoint_path = prepare_checkpoint()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from models.init_model import device, load_model, preprocess
    from models.lora import LoRAConfig
    from utils.tokenizer import tokenize

    lora_config_text = LoRAConfig(
        r=256, alpha=1024.0, bias=False, learnable_alpha=False,
        apply_q=True, apply_k=True, apply_v=True, apply_out=True,
    )
    model = load_model(
        str(checkpoint_path), model_name="ViT-B/32",
        use_oft_vision=False, use_oft_text=False,
        oft_config_vision=None, oft_config_text=None,
        use_lora_text=True, use_lora_vision=False,
        lora_config_vision=None, lora_config_text=lora_config_text,
        use_coop_text=False, use_coop_vision=False,
        precontext_length_vision=10, precontext_length_text=77,
        precontext_dropout_rate=0, pt_applied_layers=None,
    )
    model.eval()
    meta = {
        "FontCLIP_commit": FONTCLIP_COMMIT,
        "Checkpoint_SHA256": sha256_file(checkpoint_path),
        "Model": "FontCLIP / ViT-B/32 / LoRA-text checkpoint",
        "Checkpoint_source": f"Google Drive ID {FONTCLIP_CHECKPOINT_GDRIVE_ID}",
        "Prompt_method": PROMPT_METHOD,
    }
    return model, preprocess, tokenize, device, meta


def encode_scores(images, model, preprocess, tokenize, device, batch_size=8):
    import torch
    tokens = tokenize(PROMPTS).to(device)
    with torch.no_grad():
        txt = model.encode_text(tokens).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)
    all_scores, all_image_emb = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensors = [preprocess(im.convert("RGB")) for im in batch]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            feat = model.encode_image(x).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sims = feat @ txt.T
        all_image_emb.append(feat.cpu().numpy())
        all_scores.append(sims.cpu().numpy())
    return np.vstack(all_scores), np.vstack(all_image_emb)


def find_neutral_font_path() -> str:
    try:
        from matplotlib import font_manager
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        if path and Path(path).exists():
            return path
    except Exception:
        pass
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise RuntimeError("Neutral control font (DejaVu Sans) was not found in the runtime.")


def render_neutral_text(text: str, canvas=1024, target_max=800) -> Image.Image:
    text = str(text).strip() or "BRAND"
    font_path = find_neutral_font_path()
    # Fit font size deterministically to target_max while preserving line-free rendering.
    lo, hi, best = 8, 500, 80
    dummy = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(dummy)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        if max(w, h) <= target_max:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    font = ImageFont.truetype(font_path, best)
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = (canvas - w) / 2 - box[0]
    y = (canvas - h) / 2 - box[1]
    out = Image.new("L", (canvas, canvas), 255)
    ImageDraw.Draw(out).text((x, y), text, fill=0, font=font)
    return out


def wide_trait_df(meta: pd.DataFrame, scores: np.ndarray, prefix: str) -> pd.DataFrame:
    out = meta[["Filename", "Brand", "Eligibility", "Neutral_Text", "Researcher_Note"]].copy().reset_index(drop=True)
    for j, t in enumerate(TRAITS):
        out[f"{prefix}_{slug(t)}"] = scores[:, j]
    return out


def aggregate_profiles(scores: np.ndarray, meta: pd.DataFrame, prefix: str):
    """scores columns follow AAKER rows. Returns facet and 5D wide tables."""
    facet_names = AAKER["Facet"].drop_duplicates().tolist()
    facet_values = {}
    for facet in facet_names:
        idx = AAKER.index[AAKER["Facet"].eq(facet)].tolist()
        facet_values[facet] = scores[:, idx].mean(axis=1)
    facet_df = meta[["Filename", "Brand", "Eligibility", "Neutral_Text", "Researcher_Note"]].copy().reset_index(drop=True)
    for f in facet_names:
        facet_df[f"{prefix}_{slug(f)}"] = facet_values[f]

    dim_df = meta[["Filename", "Brand", "Eligibility", "Neutral_Text", "Researcher_Note"]].copy().reset_index(drop=True)
    for d in DIMENSIONS:
        facets = AAKER.loc[AAKER["Dimension"].eq(d), "Facet"].drop_duplicates().tolist()
        arr = np.vstack([facet_values[f] for f in facets]).T
        dim_df[d] = arr.mean(axis=1)
    return facet_df, dim_df


def vec_cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def vec_corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def normalized_euclidean(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.linalg.norm(a / na - b / nb))


def sign_agreement(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    # 0 is treated as no direction; exact zeros are rare after centering.
    return int(np.sum(np.sign(a) == np.sign(b)))


def plot_two_profiles(font_vals, text_vals, title):
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    fv = list(font_vals) + [font_vals[0]]
    tv = list(text_vals) + [text_vals[0]]
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, fv, linewidth=2, label="FontCLIP")
    ax.plot(angles, tv, linewidth=2, label="Text")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=8)
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    ax.set_title(title, pad=20)
    return fig


def zip_csv(files: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, df.to_csv(index=False).encode("utf-8-sig"))
    return bio.getvalue()


def load_text_profile(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = "uploaded"
    else:
        if not DEFAULT_TEXT_PROFILE_FILE.exists():
            raise FileNotFoundError(
                "기본 02 Text 5D CSV를 찾지 못했습니다. "
                "앱과 같은 GitHub 폴더에 `02_text_5D_raw_FOR_03B.csv`를 올리거나, "
                "화면의 '02 텍스트 RAW 5D CSV 교체'에서 직접 업로드하세요."
            )
        df = pd.read_csv(DEFAULT_TEXT_PROFILE_FILE)
        source = f"bundled current 02 M0_RAW: {DEFAULT_TEXT_PROFILE_FILE.name}"
    if "Method" in df.columns:
        if (df["Method"] == "M0_RAW").any():
            df = df[df["Method"] == "M0_RAW"].copy()
    missing = [c for c in ["Brand"] + DIMENSIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Text profile CSV is missing: {missing}")
    return df[[c for c in df.columns if c in (["Brand", "N_Units", "Small_Corpus_Flag", "Status"] + DIMENSIONS)]].copy(), source


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title(f"03B FontCLIP Brand Personality Analyzer · v{APP_VERSION}")
st.caption("RQ1 본분석용 · Aaker 42 traits → 15 facets → 5D → reference-centered profile → Text–FontCLIP congruence")
st.info(
    "03A의 이미지 표준화·모델·체크포인트는 유지하되, 03B에서는 `not X`와 maximin을 본측정에서 사용하지 않습니다. "
    "FontCLIP 고유 형식에 가까운 positive prompt `{trait} font`를 고정해 42개 Aaker trait를 평가합니다."
)

with st.expander("배포 리소스 상태", expanded=False):
    st.write({
        "app_dir": str(APP_DIR),
        "mapping": str(MAPPING_FILE) if MAPPING_FILE.exists() else "embedded fallback 사용",
        "03A_review": str(DEFAULT_REVIEW_FILE) if DEFAULT_REVIEW_FILE.exists() else "없음 — 수동/업로드 가능",
        "02_text_profile": str(DEFAULT_TEXT_PROFILE_FILE) if DEFAULT_TEXT_PROFILE_FILE.exists() else "없음 — 업로드 필요",
    })

with st.expander("03A → 03B 변경점", expanded=False):
    st.markdown(
        """
- 03A: 사전진단 / positive–negative contrast / maximin diversity
- 03B: **본분석 / positive prompt only / Aaker 42→15→5 / reference centering**
- raw cosine은 보존하지만 절대 심리점수로 해석하지 않음
- 최종 비교는 Text와 FontCLIP의 **relative 5D profile**
- neutral-font control은 브랜드명 문자열의 영향을 보는 민감도 분석
        """
    )

uploaded = st.file_uploader("표준화 영문 로고타입 ZIP", type=["zip"])
use_default_review = st.checkbox("03A 확정 A/B/C 검토값 자동 적용", value=True)
review_upload = st.file_uploader("다른 적격성 검토 CSV 사용 (선택)", type=["csv"], key="review_upload")
text_upload = st.file_uploader("02 텍스트 RAW 5D CSV 교체 (선택)", type=["csv"], key="text_upload")

if uploaded:
    try:
        items, skipped = safe_extract_zip(uploaded.getvalue())
    except Exception as e:
        st.exception(e); st.stop()
    if not items:
        st.error("분석 가능한 이미지가 없습니다."); st.stop()

    review = pd.DataFrame([inspect_logo(x) for x in items])
    try:
        if review_upload is not None:
            review = merge_review(review, pd.read_csv(review_upload))
            review_source = "uploaded review CSV"
        elif use_default_review and DEFAULT_REVIEW_FILE.exists():
            review = merge_review(review, pd.read_csv(DEFAULT_REVIEW_FILE))
            review_source = "bundled 03A final review"
        else:
            review_source = "manual only"
    except Exception as e:
        st.warning(f"검토 CSV 병합 실패: {e}")
        review_source = "manual only"

    st.success(f"이미지 {len(review)}개 로드 · review source: {review_source}")
    if skipped:
        st.warning(f"건너뛴 이미지 {len(skipped)}개")
        st.dataframe(pd.DataFrame(skipped), hide_index=True, use_container_width=True)

    st.subheader("1. 적격성 및 neutral text 확인")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", len(review))
    c2.metric("A", int((review.Eligibility == "A").sum()))
    c3.metric("B", int((review.Eligibility == "B").sum()))
    c4.metric("C", int((review.Eligibility == "C").sum()))

    edited = st.data_editor(
        review,
        disabled=[c for c in review.columns if c not in ["Brand", "Neutral_Text", "Eligibility", "Researcher_Note"]],
        column_config={
            "Eligibility": st.column_config.SelectboxColumn("Eligibility", options=["Unreviewed", "A", "B", "C"], required=True),
            "Neutral_Text": st.column_config.TextColumn("Neutral_Text", help="실제 로고와 동일한 브랜드 문자/대소문자를 입력. 중립서체 민감도 분석에 사용."),
        },
        use_container_width=True, height=460, hide_index=True, key="03b_review_editor"
    )

    ref_rule = st.radio(
        "Reference pool",
        ["A only (권장 본분석)", "A+B (민감도)", "All except C (탐색)"],
        horizontal=True,
    )
    if ref_rule.startswith("A only"):
        ref_mask = edited.Eligibility.eq("A")
    elif ref_rule.startswith("A+B"):
        ref_mask = edited.Eligibility.isin(["A", "B"])
    else:
        ref_mask = ~edited.Eligibility.eq("C")

    if ref_mask.sum() < 5:
        st.error("Reference pool이 너무 작습니다. 적격성 값을 확인하세요.")
        st.stop()

    do_neutral = st.checkbox("Reference-font sensitivity (DejaVu Sans) 계산", value=True)
    batch_size = st.select_slider("Batch size", options=[1, 2, 4, 8, 16], value=8)

    if st.button("03B FontCLIP 본분석 실행", type="primary"):
        try:
            with st.spinner("FontCLIP 모델 준비 중..."):
                model, preprocess, tokenize, device, model_meta = load_fontclip_runtime()
            with st.spinner(f"실제 로고 {len(items)}개 × Aaker 42 traits 분석 중..."):
                actual_images = [x["image"] for x in items]
                raw_scores, image_emb = encode_scores(actual_images, model, preprocess, tokenize, device, int(batch_size))

            neutral_scores = None
            if do_neutral:
                with st.spinner("Reference-font control 생성·분석 중..."):
                    neutral_images = [render_neutral_text(t) for t in edited["Neutral_Text"].astype(str).tolist()]
                    neutral_scores, _ = encode_scores(neutral_images, model, preprocess, tokenize, device, int(batch_size))
        except Exception as e:
            st.exception(e)
            st.stop()

        ref_idx = np.where(ref_mask.to_numpy())[0]
        trait_ref_mean = raw_scores[ref_idx].mean(axis=0)
        centered_scores = raw_scores - trait_ref_mean[None, :]

        trait_raw = wide_trait_df(edited, raw_scores, "Raw")
        trait_center = wide_trait_df(edited, centered_scores, "Centered")
        facet_raw, dim_raw = aggregate_profiles(raw_scores, edited, "Raw")
        facet_center, dim_center = aggregate_profiles(centered_scores, edited, "Centered")

        # Reference parameters
        ref_rows = []
        for j, r in AAKER.iterrows():
            ref_rows.append({
                "Dimension": r["Dimension"], "Facet": r["Facet"], "Trait": r["Trait"],
                "Prompt": PROMPTS[j], "Reference_Mean_Raw_Cosine": trait_ref_mean[j],
                "Reference_N": int(len(ref_idx)), "Reference_Rule": ref_rule,
            })
        ref_params = pd.DataFrame(ref_rows)

        # Neutral sensitivity is deliberately based on paired actual-minus-neutral raw score.
        neutral_out = pd.DataFrame()
        neutral_dim = None
        if neutral_scores is not None:
            neutral_facet, neutral_dim = aggregate_profiles(neutral_scores, edited, "NeutralRaw")
            delta = raw_scores - neutral_scores
            delta_facet, delta_dim = aggregate_profiles(delta, edited, "ActualMinusNeutral")
            neutral_out = delta_dim.copy()
            for d in DIMENSIONS:
                neutral_out[f"ActualRaw_{d}"] = dim_raw[d].to_numpy()
                neutral_out[f"NeutralRaw_{d}"] = neutral_dim[d].to_numpy()
                neutral_out[f"ActualMinusNeutral_{d}"] = delta_dim[d].to_numpy()
            neutral_out["Actual_vs_Neutral_5D_Cosine"] = [
                vec_cos(dim_raw.loc[i, DIMENSIONS], neutral_dim.loc[i, DIMENSIONS]) for i in range(len(dim_raw))
            ]

        # Text–FontCLIP congruence
        # IMPORTANT: RQ1 uses the SAME overlapping brand set as the centering reference for both models.
        # This avoids comparing profiles centered on different populations.
        congruence = pd.DataFrame()
        rq1_ref = pd.DataFrame()
        text_source = "not loaded"
        try:
            text_df, text_source = load_text_profile(text_upload)
            text_df["_key"] = text_df.Brand.map(brand_key)
            font_raw = dim_raw.copy()
            font_raw["_key"] = font_raw.Brand.map(brand_key)
            font_raw["RQ1_Eligible"] = ref_mask.to_numpy()

            merged = font_raw.merge(text_df, on="_key", suffixes=("_Font", "_Text"))
            common_ref = merged[merged["RQ1_Eligible"]].copy()
            if len(common_ref) < 5:
                raise ValueError(f"RQ1 common reference overlap is too small: {len(common_ref)}")

            font_means = {d: float(common_ref[f"{d}_Font"].mean()) for d in DIMENSIONS}
            text_means = {d: float(common_ref[f"{d}_Text"].mean()) for d in DIMENSIONS}
            rq1_ref = pd.DataFrame([
                {"Dimension": d, "FontCLIP_CommonRef_Mean": font_means[d], "Text_CommonRef_Mean": text_means[d],
                 "CommonRef_N": int(len(common_ref)), "Rule": "same overlapping eligible brand set, equal brand weights"}
                for d in DIMENSIONS
            ])

            rows = []
            for _, r in merged.iterrows():
                f = [float(r[f"{d}_Font"]) - font_means[d] for d in DIMENSIONS]
                t = [float(r[f"{d}_Text"]) - text_means[d] for d in DIMENSIONS]
                row = {
                    "Brand_Font": r["Brand_Font"],
                    "Brand_Text": r["Brand_Text"],
                    "Eligibility": r["Eligibility"],
                    "RQ1_Reference_Member": bool(r["RQ1_Eligible"]),
                    "Text_N_Units": r.get("N_Units", np.nan),
                    "Text_Small_Corpus_Flag": r.get("Small_Corpus_Flag", ""),
                    "Congruence_5D_Cosine": vec_cos(f, t),
                    "Profile_Pearson_r": vec_corr(f, t),
                    "L2_Normalized_Euclidean": normalized_euclidean(f, t),
                    "Direction_Agreement_Count_0to5": sign_agreement(f, t),
                }
                for d, fv, tv in zip(DIMENSIONS, f, t):
                    row[f"Font_{d}"] = fv
                    row[f"Text_{d}"] = tv
                rows.append(row)
            congruence = pd.DataFrame(rows).sort_values(["RQ1_Reference_Member", "Congruence_5D_Cosine"], ascending=[False, False])
        except Exception as e:
            st.warning(f"텍스트 프로파일 결합은 건너뜀: {e}")

        prompt_def = AAKER[["Item_No", "Dimension", "Facet", "Trait"]].copy()
        prompt_def["Prompt"] = PROMPTS
        prompt_def["Prompt_Role"] = "Primary positive-only FontCLIP trait prompt"

        metadata = pd.DataFrame([{
            "App_Version": APP_VERSION,
            **model_meta,
            "N_Logos": len(edited),
            "N_Reference": int(ref_mask.sum()),
            "Reference_Rule": ref_rule,
            "Aaker_Structure": "42 traits -> 15 facets -> 5 dimensions; equal weighting within each level",
            "Primary_Prompt": PROMPT_TEMPLATE,
            "Negative_Prompt_Used": False,
            "Prompt_Ensemble_Used": False,
            "Reference_Centering": "Trait raw cosine minus eligible-logo trait mean",
            "Reference_Font_Sensitivity": bool(do_neutral),
            "Reference_Font": "DejaVu Sans runtime control; no claim of psychological neutrality; font file is not bundled",
            "Text_Profile_Source": text_source,
            "N_Text_FontCLIP_Overlap": int(len(congruence)),
            "N_RQ1_Common_Reference": int(rq1_ref["CommonRef_N"].iloc[0]) if not rq1_ref.empty else 0,
            "RQ1_Centering": "dimension-wise centering within the same overlapping eligible brand set for both Text and FontCLIP",
            "Primary_RQ1_Metric": "cosine similarity between common-reference-centered 5D profiles",
        }])

        # Embeddings retained for reproducibility, not used as primary BP output.
        emb_cols = [f"Embedding_{i:03d}" for i in range(image_emb.shape[1])]
        embeddings = pd.DataFrame(image_emb, columns=emb_cols)
        embeddings.insert(0, "Brand", edited.Brand.astype(str).tolist())
        embeddings.insert(0, "Filename", edited.Filename.astype(str).tolist())

        st.session_state["03B_R"] = {
            "review": edited.copy(), "trait_raw": trait_raw, "trait_center": trait_center,
            "facet_raw": facet_raw, "facet_center": facet_center,
            "dim_raw": dim_raw, "dim_center": dim_center,
            "neutral": neutral_out, "congruence": congruence,
            "ref_params": ref_params, "rq1_ref": rq1_ref, "prompt_def": prompt_def,
            "metadata": metadata, "embeddings": embeddings,
        }
        st.success("03B 본분석 완료")

if "03B_R" in st.session_state:
    R = st.session_state["03B_R"]
    st.divider()
    st.header("03B 결과")

    dimc = R["dim_center"].copy()
    st.subheader("2. FontCLIP reference-centered 5D")
    st.dataframe(dimc[["Brand", "Eligibility"] + DIMENSIONS], hide_index=True, use_container_width=True)
    st.caption("0보다 크면 현재 reference pool 평균보다 상대적으로 높은 방향, 0보다 작으면 상대적으로 낮은 방향입니다. 절대 심리점수가 아닙니다.")

    if not R["congruence"].empty:
        st.subheader("3. Text ↔ FontCLIP RQ1 congruence")
        st.dataframe(
            R["congruence"][["Brand_Font", "Eligibility", "Text_N_Units", "Congruence_5D_Cosine", "Profile_Pearson_r", "L2_Normalized_Euclidean", "Direction_Agreement_Count_0to5"]],
            hide_index=True, use_container_width=True
        )
        st.caption("두 모델은 같은 overlapping eligible brand set을 기준으로 각각 center한 뒤 비교합니다. Cosine이 주지표이며, Pearson r은 5개 차원만으로 계산되므로 보조적 기술값으로만 사용합니다.")
        b = st.selectbox("프로파일 비교 브랜드", R["congruence"]["Brand_Font"].tolist())
        rr = R["congruence"][R["congruence"]["Brand_Font"] == b].iloc[0]
        fv = [float(rr[f"Font_{d}"]) for d in DIMENSIONS]
        tv = [float(rr[f"Text_{d}"]) for d in DIMENSIONS]
        st.pyplot(plot_two_profiles(fv, tv, f"{b} · Text vs FontCLIP"), use_container_width=False)

    if not R["neutral"].empty:
        st.subheader("4. Reference-font sensitivity")
        cols = ["Brand", "Eligibility", "Actual_vs_Neutral_5D_Cosine"] + [f"ActualMinusNeutral_{d}" for d in DIMENSIONS]
        st.dataframe(R["neutral"][cols], hide_index=True, use_container_width=True)
        st.caption("Actual−Reference는 동일 문자열을 DejaVu Sans 기준서체로 렌더링했을 때와의 차이입니다. DejaVu Sans를 심리적으로 중립적이라고 가정하지 않으며, 문자열/서체 영향 가능성을 확인하는 민감도 분석입니다.")

    files = {
        "03B_01_logo_review.csv": R["review"],
        "03B_02_prompt_definition.csv": R["prompt_def"],
        "03B_03_trait_scores_raw.csv": R["trait_raw"],
        "03B_04_trait_scores_reference_centered.csv": R["trait_center"],
        "03B_05_facet_profiles_raw.csv": R["facet_raw"],
        "03B_06_facet_profiles_reference_centered.csv": R["facet_center"],
        "03B_07_5D_profiles_raw.csv": R["dim_raw"],
        "03B_08_5D_profiles_reference_centered.csv": R["dim_center"],
        "03B_09_reference_font_sensitivity.csv": R["neutral"],
        "03B_10_text_fontclip_congruence.csv": R["congruence"],
        "03B_11_reference_parameters.csv": R["ref_params"],
        "03B_12_RQ1_common_reference_parameters.csv": R["rq1_ref"],
        "03B_13_image_embeddings.csv": R["embeddings"],
        "03B_14_run_metadata.csv": R["metadata"],
    }
    st.subheader("5. 결과 다운로드")
    st.download_button(
        "03B 전체 CSV ZIP 다운로드", zip_csv(files),
        file_name="03B_fontclip_brand_personality_results_v1_1.zip",
        mime="application/zip", type="primary"
    )
    for i in range(0, len(files), 3):
        cs = st.columns(3)
        for c, (name, df) in zip(cs, list(files.items())[i:i+3]):
            with c:
                st.download_button(name.replace(".csv", ""), df.to_csv(index=False).encode("utf-8-sig"), name, "text/csv", use_container_width=True)

st.divider()
st.caption("03B는 영문 로고타입의 FontCLIP 본분석입니다. 03A 사전진단 결과는 방법개발 근거로 보존하며, 04 CLIP 접점분석과 05 인간평가는 별도 단계에서 수행합니다.")
