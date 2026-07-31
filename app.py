import base64
import json
import socket
import time
from io import BytesIO
from pathlib import Path


import numpy as np
import pandas as pd
import qrcode
import streamlit as st

st.set_page_config(page_title="EMA Demo Viewer Ver.0.2.2", page_icon="EMA", layout="wide", initial_sidebar_state="collapsed")

TIMELINE_EVENTS = [
    ("00:00", "Ready Position", "Head and jaw alignment detected", "#4dd0e1"),
    ("00:03", "Mouth Opening", "Right hand support begins", "#7ce38b"),
    ("00:06", "Blade Insertion", "Exploration path slightly wide", "#ffd166"),
    ("00:08", "Vallecula Search", "Peak coordination demand", "#ff6b6b"),
    ("00:11", "Tube Delivery", "Trajectory recovered", "#a78bfa"),
]

REPORT_BEFORE = [
    ("頭位保持", 82, "頭部後屈の保持は安定しています。探索中の微小な戻りを抑えると視野がさらに安定します。"),
    ("開口戦略", 76, "右手の開口補助がやや遅れ、ブレード挿入開始との同期に余地があります。"),
    ("探索効率", 69, "喉頭蓋谷到達までの探索経路が少し広く、目標点への収束に時間を使っています。"),
    ("左右協調", 78, "左手のブレード操作と右手の補助は概ね連動していますが、ピーク局面で左右差が出ています。"),
    ("筋活動効率", 73, "肩周囲の活動が一時的に高まり、細かな操作を上肢全体で補っている傾向があります。"),
    ("チューブ操作", 81, "挿入角度は安定しています。視野確保後のチューブ移行を少し早められます。"),
]

REPORT_AFTER = [
    ("頭位保持", 88, "頭部後屈と下顎挙上の保持が安定し、視野形成の土台が作れています。"),
    ("開口戦略", 84, "右手の開口補助と左手の挿入開始が近づき、運動の立ち上がりが滑らかです。"),
    ("探索効率", 77, "喉頭蓋谷への探索経路が短くなり、ブレード先端の迷いが減っています。"),
    ("左右協調", 85, "左右の役割分担が明確で、開口補助とブレード操作が連続動作としてつながっています。"),
    ("筋活動効率", 80, "肩の過活動が抑えられ、手首と前腕中心の細かな制御に移行しています。"),
    ("チューブ操作", 87, "視野確保からチューブ誘導までの切り替えが速く、操作の連続性が高まっています。"),
]

COACHING_BEFORE = {
    "Good Point": "頭位保持と開口動作の同期は良好です。初期姿勢から挿入準備までの流れは安定しています。",
    "Improvement Point": "喉頭蓋谷到達までの探索時間がやや長くなっています。ブレード先端の移動量を小さく保つ意識が有効です。",
    "Training Advice": "次回は右手の開口補助と左手のブレード操作を同時に開始しましょう。2秒停止を入れて視野が崩れない位置を確認してください。",
    "Expert Strategy Comparison": "熟練者は頭部後屈、下顎挙上、ブレード挿入を連続した一つの運動戦略として行います。",
}

COACHING_AFTER = {
    "Good Point": "頭位保持、開口補助、ブレード挿入の時間差が小さくなり、視野形成までの運動がまとまっています。",
    "Improvement Point": "ピーク局面で右肩の活動がまだ少し上がります。肩ではなく手首と前腕で角度を微調整しましょう。",
    "Training Advice": "下顎挙上を先に固定し、左手のブレード先端を短い軌道で進める反復を3セット行ってください。",
    "Expert Strategy Comparison": "熟練者は喉頭蓋谷を探す前に、頭位と開口で視野の入口を作ります。探索ではなく誘導に近い運動戦略です。",
}

ANALYSIS_STEPS = ["Capturing Motion", "Detecting Keypoints", "Estimating Motor Strategy", "Generating Coaching Report"]
SUBJECT_JSON_PATH = Path(__file__).parent / "demo_data" / "public_sample" / "subject.json"
LOGO_IMAGE_PATH = Path(__file__).parent / "assets" / "ema_logo.png"



def load_subject_metadata() -> dict:
    with SUBJECT_JSON_PATH.open("r", encoding="utf-8-sig") as subject_file:
        return json.load(subject_file)

def render_subject_information() -> None:
    st.markdown('<div class="connection-panel"><div class="connection-title">Subject Information</div>', unsafe_allow_html=True)
    try:
        subject = load_subject_metadata()
    except FileNotFoundError:
        st.error(f"Subject metadata not found: {SUBJECT_JSON_PATH}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    except json.JSONDecodeError as error:
        st.error(f"Subject metadata JSON is invalid: {error}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    rows = [
        ("Subject ID", subject.get("id", "")),
        ("Name", subject.get("name", "")),
        ("Group", subject.get("group", "")),
        ("Front Videos", ", ".join(subject.get("front_videos", []))),
        ("Side Videos", ", ".join(subject.get("side_videos", []))),
        ("EMG Files", ", ".join(subject.get("emg_files", []))),
    ]
    st.table(pd.DataFrame(rows, columns=["Field", "Value"]))
    st.markdown('</div>', unsafe_allow_html=True)

def build_emg_data(analyzed: bool) -> pd.DataFrame:
    rng = np.random.default_rng(84 if analyzed else 42)
    t = np.linspace(0, 12, 360)
    right = 0.42 + 0.18 * np.sin(t * 2.3) + rng.normal(0, 0.025, len(t))
    jaw = 0.34 + 0.14 * np.sin(t * 2.9 + 0.8) + rng.normal(0, 0.02, len(t))
    shoulder = 0.28 + 0.09 * np.sin(t * 3.4 + 1.7) + rng.normal(0, 0.018, len(t))
    burst = np.exp(-0.5 * ((t - 7.4) / 0.45) ** 2)
    right += burst * (0.23 if analyzed else 0.28)
    jaw += burst * (0.16 if analyzed else 0.19)
    shoulder += np.exp(-0.5 * ((t - 8.2) / 0.38) ** 2) * (0.08 if analyzed else 0.12)
    return pd.DataFrame({
        "Time (s)": t,
        "Right Hand EMG": np.clip(right, 0, 1),
        "Jaw Support EMG": np.clip(jaw, 0, 1),
        "Shoulder Stabilizer EMG": np.clip(shoulder, 0, 1),
    })


def css() -> None:
    st.markdown("""
    <style>
    :root{--panel:rgba(20,24,34,.88);--soft:rgba(255,255,255,.055);--stroke:rgba(255,255,255,.12);--text:#f6f7fb;--muted:#9aa3b2;--cyan:#4dd0e1;--green:#7ce38b;--yellow:#ffd166;--red:#ff6b6b;--blue:#0b75d1;--navy:#071424;--ink:#102033;--paper:#f7fbff}
    .stApp{color:var(--text);background:linear-gradient(135deg,#03050a 0%,#101524 58%,#07090f 100%)}
    [data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none}.block-container{max-width:1360px;padding:22px 24px 34px}.ipad-shell{border:1px solid rgba(255,255,255,.16);border-radius:34px;padding:18px;background:linear-gradient(145deg,rgba(255,255,255,.16),rgba(255,255,255,.04));box-shadow:0 26px 80px rgba(0,0,0,.44),inset 0 0 0 1px rgba(255,255,255,.05)}
    .screen{min-height:calc(100vh - 92px);border-radius:24px;padding:26px;background:rgba(7,9,15,.84);border:1px solid rgba(255,255,255,.11);overflow:hidden}.phone-warning{display:none;margin-bottom:12px;padding:10px 12px;border-radius:12px;color:#071018;background:var(--yellow);font-weight:800}
    .splash{min-height:calc(100vh - 88px);display:flex;align-items:center;justify-content:center;border-radius:28px;border:1px solid rgba(255,255,255,.11);background:radial-gradient(circle at 50% 18%,rgba(77,208,225,.12),transparent 34%),linear-gradient(145deg,#05070d 0%,#0b1019 54%,#020305 100%);box-shadow:inset 0 0 70px rgba(255,255,255,.025),0 26px 80px rgba(0,0,0,.44)}
    .splash-inner{text-align:center;width:min(820px,92vw);padding:64px 34px 110px}.logo-wrap{display:flex;justify-content:center}.logo-img{display:block;width:auto;object-fit:contain}.splash-logo-img{max-width:min(320px,72vw);max-height:180px;margin:0 auto 26px}.home-logo-img{max-width:118px;max-height:78px;margin:0 auto 16px}.logo-mark{display:inline-grid;place-items:center;border-radius:24px;background:linear-gradient(145deg,#0d2d4f,#0b75d1);color:#fff;font-weight:900;letter-spacing:.08em;box-shadow:0 18px 36px rgba(15,89,158,.18)}.splash-logo{width:min(260px,64vw);height:min(150px,36vw);margin:0 auto 26px;font-size:72px}.home-logo{width:86px;height:86px;margin:0 auto 18px;font-size:30px}.splash-subtitle{margin-top:18px;color:#e5edf7;font-size:25px;font-weight:750}.splash-system{margin-top:16px;color:#aeb8c7;font-size:18px;font-weight:650}.splash-copy{margin-top:24px;color:#4dd0e1;font-size:16px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
    .start-button-wrap div[data-testid="stButton"]{display:flex;justify-content:center;margin-top:-106px}.start-button-wrap button,.nav-home button,.nav-back button{min-height:56px;border-radius:999px;border:1px solid rgba(77,208,225,.42);background:linear-gradient(135deg,#0d7bdd,#0b4c96);color:white;font-size:16px;font-weight:900;letter-spacing:.08em;box-shadow:0 12px 26px rgba(13,92,172,.18)}.stButton button{border-radius:999px;border:1px solid rgba(255,255,255,.22);background:linear-gradient(90deg,var(--cyan),var(--green));color:#061018;font-weight:850;box-shadow:0 12px 32px rgba(77,208,225,.24)}
    .home-shell{min-height:calc(100vh - 86px);padding:42px;border-radius:28px;background:linear-gradient(180deg,#ffffff 0%,#f4f8fd 100%);color:var(--ink);border:1px solid rgba(12,28,48,.08);box-shadow:0 24px 70px rgba(3,10,22,.28)}.home-hero{text-align:center;margin:24px auto 42px;max-width:900px}.home-hero h1{margin:0;color:#061426;font-size:56px;line-height:1;font-weight:900;letter-spacing:.04em}.home-hero .subtitle{margin-top:12px;color:#1c344d;font-size:23px;font-weight:760}.home-hero .system{margin-top:10px;color:#52677e;font-size:17px}.home-hero .copy{margin-top:18px;color:#0a6ecb;font-size:13px;font-weight:850;letter-spacing:.12em;text-transform:uppercase}
    .home-card{height:274px;padding:30px;border-radius:22px;background:linear-gradient(180deg,#fff 0%,#fbfdff 100%);border:1px solid rgba(12,28,48,.1);box-shadow:0 18px 44px rgba(11,35,68,.1);display:flex;flex-direction:column;justify-content:flex-start}.home-card .kicker{color:#0b75d1;font-size:12px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}.home-card h2{margin:17px 0 12px;color:#102033;font-size:25px;line-height:1.2}.home-card p{margin:0;color:#607187;font-size:15px;line-height:1.75;white-space:pre-line}.card-action div[data-testid="stButton"]{margin-top:-84px;padding:0 22px}.card-action button{min-height:54px;border-radius:16px;background:linear-gradient(135deg,#0d7bdd,#0b4c96);color:white;font-weight:850;box-shadow:0 12px 26px rgba(13,92,172,.18)}.about-ema{margin:54px auto 0;max-width:820px;padding:28px 30px;border-radius:22px;background:#fff;border:1px solid rgba(12,28,48,.08);box-shadow:0 16px 38px rgba(11,35,68,.08);text-align:left}.about-ema h2{margin:0 0 14px;color:#102033;font-size:22px}.about-ema strong{display:block;margin-bottom:8px;color:#1c344d}.about-ema p{margin:0;color:#66778c;font-size:14px;line-height:1.75}.home-footer{margin-top:34px;text-align:center;color:#7b8ca1;font-size:12px;line-height:1.8}
    .app-nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}.nav-brand{color:#dbe8f2;font-size:15px;font-weight:850;letter-spacing:.08em}.nav-back div[data-testid="stButton"]{display:flex;justify-content:flex-end}.nav-home button,.nav-back button{min-width:142px;box-shadow:none}.coming-soon,.loading-screen{min-height:calc(100vh - 120px);display:grid;place-items:center;text-align:center;border-radius:28px;background:linear-gradient(180deg,#f8fbff,#edf4fb);color:#102033;border:1px solid rgba(12,28,48,.08)}.coming-soon .icon{font-size:50px;margin-bottom:14px}.coming-soon h1,.loading-screen h1{margin:0;font-size:44px;color:#102033}.coming-soon p,.loading-screen p{margin:12px 0 0;color:#607187;font-size:18px}.coming-soon .version{margin-top:8px;color:#0b75d1;font-size:13px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}.loading-dot{width:46px;height:46px;margin:0 auto 20px;border-radius:999px;border:4px solid rgba(11,117,209,.16);border-top-color:#0b75d1}.dev-info{margin-bottom:16px}.dev-info div[data-testid="stExpander"]{border:1px solid rgba(255,255,255,.12);border-radius:18px;background:rgba(20,24,34,.72)}
    .topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:18px}.brand h1{margin:0;font-size:54px;line-height:.95}.brand .subtitle{margin-top:8px;color:var(--muted);font-size:17px}.tagline{align-self:center;padding:10px 14px;border-radius:999px;color:#061018;font-weight:750;background:linear-gradient(90deg,var(--cyan),var(--green));white-space:nowrap}
    div[data-testid="stVerticalBlock"]>div:has(.panel-title){padding:16px;border:1px solid var(--stroke);border-radius:18px;background:var(--panel);box-shadow:0 16px 40px rgba(0,0,0,.24)}.panel-title{display:flex;align-items:center;justify-content:space-between;margin:0 0 12px;font-size:16px;font-weight:750}.panel-title span{color:var(--muted);font-size:12px;font-weight:650}.mode-note{padding:12px 14px;border:1px solid var(--stroke);border-radius:14px;background:var(--soft);color:#d8deea;font-size:13px;line-height:1.45}
    .video-area{position:relative;min-height:414px;border-radius:18px;overflow:hidden;border:1px solid rgba(255,255,255,.13);background:linear-gradient(0deg,rgba(0,0,0,.42),rgba(0,0,0,.06)),linear-gradient(135deg,#161d28,#253041 48%,#111620);box-shadow:inset 0 0 60px rgba(0,0,0,.34)}.video-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.06) 1px,transparent 1px);background-size:48px 48px;opacity:.32}.video-placeholder{position:absolute;inset:0;display:grid;place-items:center;color:rgba(246,247,251,.78);font-size:14px}.overlay-label{position:absolute;top:14px;left:14px;padding:7px 10px;border-radius:999px;color:#061018;background:rgba(77,208,225,.92);font-size:12px;font-weight:850;z-index:5}.mocap{position:absolute;left:50%;top:52%;width:286px;height:300px;transform:translate(-50%,-50%);z-index:4}.joint,.segment{position:absolute;background:var(--cyan);box-shadow:0 0 20px rgba(77,208,225,.65)}.joint{width:15px;height:15px;border-radius:50%;transform:translate(-50%,-50%);border:1px solid rgba(255,255,255,.62)}.joint.target{width:19px;height:19px;background:var(--green);box-shadow:0 0 26px rgba(124,227,139,.8)}.joint.hot{background:var(--yellow);box-shadow:0 0 26px rgba(255,209,102,.8)}.segment{height:4px;border-radius:999px;transform-origin:left center;opacity:.9}.video-hud{position:absolute;left:18px;right:18px;bottom:18px;display:flex;justify-content:space-between;gap:12px;z-index:6}.hud-chip{padding:8px 11px;border-radius:999px;background:rgba(0,0,0,.48);border:1px solid rgba(255,255,255,.12);font-size:13px}
    .timeline-item{display:grid;grid-template-columns:58px 10px 1fr;gap:12px;align-items:start;padding:0 0 15px}.timeline-time{color:var(--muted);font-size:13px}.timeline-dot{width:10px;height:10px;margin-top:4px;border-radius:50%;box-shadow:0 0 16px currentColor}.timeline-label{font-weight:700;line-height:1.15}.timeline-state{color:var(--muted);font-size:13px;margin-top:3px}.score-card,.coach-card{padding:12px;border:1px solid var(--stroke);border-radius:14px;background:var(--soft);margin-bottom:10px}.score-row{display:flex;justify-content:space-between;gap:12px;margin-bottom:8px;font-size:14px}.score-value{color:var(--green);font-weight:800}.score-comment{margin-top:8px;color:#cbd5e1;font-size:12.5px;line-height:1.45}.bar{height:8px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.1)}.bar span{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--cyan),var(--green))}.coach-card strong{display:block;margin-bottom:6px;color:var(--cyan)}.coach-card p{margin:0;color:#d8deea;font-size:14px;line-height:1.48}
    .connection-panel{margin-bottom:16px;padding:16px;border:1px solid var(--stroke);border-radius:18px;background:var(--panel);box-shadow:0 16px 40px rgba(0,0,0,.24)}.connection-title{font-size:18px;font-weight:850;margin-bottom:10px}.connection-note{color:#d8deea;font-size:13px;line-height:1.45;margin-top:8px}.connection-label{color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
    @media(max-width:760px){.phone-warning{display:block}.block-container{min-width:0;padding:10px}.ipad-shell{border-radius:24px;padding:10px}.screen{border-radius:18px;padding:16px}.home-shell{min-width:0;padding:24px 18px}.home-hero{margin-top:20px}.home-hero h1{font-size:42px}.home-hero .subtitle{font-size:20px}.home-card{height:auto;min-height:230px}.about-ema{margin-top:42px;padding:22px}.splash-inner{padding-inline:20px}.splash-subtitle{font-size:21px}.splash-system{font-size:16px}.start-button-wrap div[data-testid="stButton"]{margin-top:-96px}.app-nav{gap:10px}.nav-brand{display:none}.nav-home button,.nav-back button{min-width:120px}}
    </style>
    """, unsafe_allow_html=True)

def get_lan_ipv4() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_address = sock.getsockname()[0]
            if ip_address and not ip_address.startswith("127."):
                return ip_address
    except OSError:
        pass

    try:
        hostname = socket.gethostname()
        for candidate in socket.gethostbyname_ex(hostname)[2]:
            if candidate and not candidate.startswith("127."):
                return candidate
    except OSError:
        pass

    return "127.0.0.1"


def build_qr_code(url: str) -> BytesIO:
    qr = qrcode.QRCode(version=1, box_size=8, border=3)
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="#061018", back_color="#ffffff")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def render_connection_information() -> None:
    local_url = "http://localhost:8501"
    network_url = f"http://{get_lan_ipv4()}:8501"

    st.markdown('<div class="connection-panel"><div class="connection-title">Connection Information</div>', unsafe_allow_html=True)
    local_col, network_col, qr_col = st.columns([1, 1, 0.62], gap="large")
    with local_col:
        st.markdown('<div class="connection-label">Local</div>', unsafe_allow_html=True)
        st.code(local_url, language=None)
    with network_col:
        st.markdown('<div class="connection-label">Network</div>', unsafe_allow_html=True)
        st.code(network_url, language=None)
    with qr_col:
        st.image(build_qr_code(network_url), caption="Network URL QR", width=150)
    st.markdown('<div class="connection-note">Connect your iPad or smartphone to the same Wi-Fi network.</div></div>', unsafe_allow_html=True)


def render_external_demo_access() -> None:
    st.sidebar.markdown("### External Demo Access")
    st.sidebar.code("cloudflared tunnel --url http://localhost:8501", language="powershell")
    st.sidebar.markdown(
        "If your iPad or smartphone is not on the same Wi-Fi network, use Cloudflare Tunnel. "
        "Then open the generated trycloudflare.com URL."
    )
    st.sidebar.warning(
        "Temporary demo access only. Do not expose real patient, student, or research data.\n\n"
        "一時的なデモ公開専用です。実在の患者情報、学生個人情報、研究データは公開しないでください。"
    )

    st.markdown('<div class="connection-panel"><div class="connection-title">External Demo Access</div>', unsafe_allow_html=True)
    st.markdown(
        "If your iPad or smartphone is not on the same Wi-Fi network, use Cloudflare Tunnel:"
    )
    st.code("cloudflared tunnel --url http://localhost:8501", language="powershell")
    st.markdown("Then open the generated `trycloudflare.com` URL.")
    st.warning(
        "This is for temporary demo access only. Do not expose real patient, student, or research data.\n\n"
        "一時的なデモ公開専用です。実在の患者情報、学生個人情報、未公開研究データは表示しないでください。"
    )

    external_url = st.text_input(
        "External Demo URL",
        placeholder="https://xxxx.trycloudflare.com",
        help="Paste the Cloudflare Tunnel URL here to generate a QR code for phones and iPads.",
    ).strip()

    if external_url:
        if external_url.startswith(("https://", "http://")):
            st.code(external_url, language=None)
            st.image(build_qr_code(external_url), caption="External Demo URL QR", width=190)
        else:
            st.error("Please enter a URL starting with https:// or http://")
    st.markdown('</div>', unsafe_allow_html=True)

def overlay_html() -> str:
    return """
    <div class="overlay-label">Demo Overlay - not real analysis</div><div class="mocap">
    <div class="joint" style="left:50%;top:8%"></div><div class="joint" style="left:35%;top:28%"></div><div class="joint" style="left:65%;top:28%"></div><div class="joint" style="left:25%;top:47%"></div><div class="joint" style="left:76%;top:47%"></div><div class="joint hot" style="left:20%;top:66%"></div><div class="joint hot" style="left:84%;top:61%"></div><div class="joint target" style="left:63%;top:42%"></div><div class="joint" style="left:16%;top:71%"></div><div class="joint" style="left:88%;top:66%"></div>
    <div class="segment" style="left:50%;top:10%;width:67px;transform:rotate(121deg)"></div><div class="segment" style="left:50%;top:10%;width:67px;transform:rotate(59deg)"></div><div class="segment" style="left:36%;top:29%;width:67px;transform:rotate(108deg)"></div><div class="segment" style="left:65%;top:29%;width:66px;transform:rotate(72deg)"></div><div class="segment" style="left:25%;top:48%;width:60px;transform:rotate(100deg)"></div><div class="segment" style="left:76%;top:48%;width:56px;transform:rotate(73deg)"></div><div class="segment" style="left:64%;top:42%;width:67px;transform:rotate(22deg);background:#7ce38b"></div><div class="segment" style="left:36%;top:29%;width:86px;transform:rotate(0deg)"></div></div>
    """


def render_demo_video(show_overlay: bool, label: str) -> None:
    overlay = overlay_html() if show_overlay else ""
    st.markdown(f"""
    <div class="video-area"><div class="video-grid"></div><div class="video-placeholder">{label}</div>{overlay}
    <div class="video-hud"><div class="hud-chip">Pose Confidence DEMO</div><div class="hud-chip">Vallecula Search 00:08</div><div class="hud-chip">Coach Mode ON</div></div></div>
    """, unsafe_allow_html=True)


def logo_markup(context: str) -> str:
    if LOGO_IMAGE_PATH.exists():
        image_data = base64.b64encode(LOGO_IMAGE_PATH.read_bytes()).decode("ascii")
        css_class = "splash-logo-img" if context == "splash" else "home-logo-img"
        return f'<div class="logo-wrap"><img class="logo-img {css_class}" src="data:image/png;base64,{image_data}" alt="EMA logo"></div>'

    css_class = "splash-logo" if context == "splash" else "home-logo"
    return f'<div class="logo-wrap"><div class="logo-mark {css_class}">EMA</div></div>'


def go_home() -> None:
    st.session_state.view = "home"


def render_app_nav(show_back: bool = False) -> None:
    left, center, right = st.columns([1, 2, 1])
    with left:
        st.markdown('<div class="nav-home">', unsafe_allow_html=True)
        if st.button("EMA HOME", width="stretch"):
            go_home()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with center:
        st.markdown('<div class="nav-brand">EMA / Expert Motion Coaching AI</div>', unsafe_allow_html=True)
    with right:
        if show_back:
            st.markdown('<div class="nav-back">', unsafe_allow_html=True)
            if st.button("BACK", width="stretch"):
                go_home()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def start_screen() -> None:
    st.markdown('<div class="ipad-shell"><div class="splash"><div class="splash-inner">', unsafe_allow_html=True)
    st.markdown(logo_markup("splash"), unsafe_allow_html=True)
    st.markdown("""
        <div class="splash-subtitle">Expert Motion Coaching AI</div>
        <div class="splash-system">AI-powered Endotracheal Intubation Coaching System</div>
        <div class="splash-copy">From Motion to Mastery</div>
        </div></div></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="start-button-wrap">', unsafe_allow_html=True)
    if st.button("START", width=240):
        st.session_state.view = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_home_card(kicker: str, title: str, body: str, button_label: str) -> bool:
    st.markdown(f'<div class="home-card"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{body}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-action">', unsafe_allow_html=True)
    clicked = st.button(button_label, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    return clicked


def home_screen() -> None:
    st.markdown('<div class="ipad-shell"><div class="home-shell">', unsafe_allow_html=True)
    st.markdown(logo_markup("home"), unsafe_allow_html=True)
    st.markdown("""
        <div class="home-hero">
            <h1>EMA</h1>
            <div class="subtitle">Expert Motion Coaching AI</div>
            <div class="system">AI-powered Endotracheal Intubation Coaching System</div>
            <div class="copy">From Motion to Mastery</div>
        </div>
    """, unsafe_allow_html=True)
    teaching, recording, demo = st.columns(3, gap="large")
    with teaching:
        if render_home_card("Teaching Mode", "AI-guided Skill Coaching", "\u719f\u7df4\u8005\u30e2\u30c7\u30eb\u306b\u3088\u308b\\n\u30ea\u30a2\u30eb\u30bf\u30a4\u30e0\u6559\u80b2", "TEACHING"):
            st.session_state.view = "coming_soon"
            st.rerun()
    with recording:
        if render_home_card("Recording & Assessment", "Motion Analysis", "\u52d5\u753b\u89e3\u6790\\nAI\u63a1\u70b9\\n\u8a55\u4fa1\u30ec\u30dd\u30fc\u30c8", "RECORDING"):
            st.session_state.view = "coming_soon"
            st.rerun()
    with demo:
        if render_home_card("Demo Viewer", "Current Public Demo", "EMA\u30c7\u30e2\u3092\u4f53\u9a13", "DEMO VIEWER"):
            st.session_state.view = "loading_dashboard"
            st.rerun()
    st.markdown("""
        <div class="about-ema">
            <h2>About EMA</h2>
            <strong>Expert Motion Coaching AI</strong>
            <p>AI-powered Endotracheal Intubation Coaching System</p>
            <p>An educational platform designed to support objective skill assessment and AI-assisted coaching using motion analysis.</p>
        </div>
        <div class="home-footer">Version 0.5<br>&copy; 2026 EMA Project</div>
    """, unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


def coming_soon_screen() -> None:
    render_app_nav(show_back=True)
    st.markdown("""
        <div class="coming-soon">
            <div>
                <div class="icon">&#128679;</div>
                <h1>Coming Soon</h1>
                <p>AI Coaching Module</p>
                <div class="version">Version 1.0</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def loading_screen() -> None:
    st.markdown("""
        <div class="loading-screen">
            <div>
                <div class="loading-dot"></div>
                <h1>Loading EMA Demo...</h1>
                <p>Preparing the public demonstration</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(0.7)
    st.session_state.view = "dashboard"
    st.rerun()


def analyze() -> None:
    progress = st.progress(0)
    status = st.empty()
    for i, step in enumerate(ANALYSIS_STEPS, start=1):
        status.markdown(f"**{step}**")
        progress.progress(i / len(ANALYSIS_STEPS))
        time.sleep(0.45)
    status.success("Coaching Report Generated")
    st.session_state.analyzed = True


def render_report(items: list[tuple[str, int, str]]) -> None:
    for label, score, comment in items:
        st.markdown(f"""<div class="score-card"><div class="score-row"><span>{label}</span><span class="score-value">{score}</span></div><div class="bar"><span style="width:{score}%"></span></div><div class="score-comment">{comment}</div></div>""", unsafe_allow_html=True)


def render_coaching(items: dict[str, str]) -> None:
    for title, body in items.items():
        st.markdown(f"""<div class="coach-card"><strong>{title}</strong><p>{body}</p></div>""", unsafe_allow_html=True)


def dashboard() -> None:
    analyzed = st.session_state.analyzed
    report = REPORT_AFTER if analyzed else REPORT_BEFORE
    coaching = COACHING_AFTER if analyzed else COACHING_BEFORE
    emg = build_emg_data(analyzed)

    st.markdown('<div class="ipad-shell"><div class="screen">', unsafe_allow_html=True)
    st.markdown('<div class="phone-warning">iPad横向きでの利用を推奨します</div>', unsafe_allow_html=True)
    st.markdown('<div class="topbar"><div class="brand"><h1>EMA</h1><div class="subtitle">Expert Motion Coaching AI</div></div><div class="tagline">From Motion to Mastery</div></div>', unsafe_allow_html=True)

    controls, toggles = st.columns([1.4, 1], gap="large")
    with controls:
        mode = st.radio("Input Mode", ["Demo Mode", "Camera Mode", "Video Upload"], horizontal=True, label_visibility="collapsed")
    with toggles:
        overlay_on = st.toggle("Motion Capture Demo Overlay", value=True)

    left, right = st.columns([1.72, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Camera / Video Area <span>Demo Session 00:12</span></div>', unsafe_allow_html=True)
        if mode == "Demo Mode":
            render_demo_video(overlay_on, "Demo video placeholder / simulated intubation motion")
        elif mode == "Camera Mode":
            st.markdown('<div class="mode-note">Camera Mode uses Streamlit camera_input. Browser and device permissions may limit camera access. Captured images are not analyzed.</div>', unsafe_allow_html=True)
            image = st.camera_input("Camera capture", label_visibility="collapsed")
            if image is None:
                render_demo_video(overlay_on, "Camera unavailable or not permitted. Showing demo placeholder.")
            else:
                st.image(image, caption="Camera capture preview - demo only", width="stretch")
                render_demo_video(overlay_on, "Demo Overlay preview for captured frame")
        else:
            video = st.file_uploader("Upload a demo video", type=["mp4", "mov", "m4v", "avi"])
            if video is None:
                render_demo_video(overlay_on, "Upload a video or use the simulated demo area")
            else:
                st.video(video)
                if overlay_on:
                    render_demo_video(True, "Demo Overlay preview for uploaded video")

        bcol, scol = st.columns([0.34, 0.66], gap="large")
        with bcol:
            if st.button("Analyze Motion", width="stretch"):
                analyze()
                st.rerun()
        with scol:
            msg = "Report updated from demo analysis" if analyzed else "Ready for demo analysis"
            st.markdown(f'<div class="mode-note">{msg}</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-title">EMG Waveform <span>Dummy Sensor Data</span></div>', unsafe_allow_html=True)
        st.line_chart(emg, x="Time (s)", y=["Right Hand EMG", "Jaw Support EMG", "Shoulder Stabilizer EMG"], height=250, color=["#4dd0e1", "#7ce38b", "#ffd166"])

    with right:
        st.markdown('<div class="panel-title">Event Timeline <span>Live Tags</span></div>', unsafe_allow_html=True)
        for t, label, state, accent in TIMELINE_EVENTS:
            st.markdown(f'<div class="timeline-item"><div class="timeline-time">{t}</div><div class="timeline-dot" style="color:{accent};background:{accent}"></div><div><div class="timeline-label">{label}</div><div class="timeline-state">{state}</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Motion Report <span>Score / 100</span></div>', unsafe_allow_html=True)
        render_report(report)

    lower_left, lower_right = st.columns([1, 1], gap="large")
    with lower_left:
        st.markdown('<div class="panel-title">EMG Detail <span>Last 8 Samples</span></div>', unsafe_allow_html=True)
        st.dataframe(emg.tail(8).round(3), width="stretch", hide_index=True)
    with lower_right:
        st.markdown('<div class="panel-title">AI Coaching <span>EMA Insight</span></div>', unsafe_allow_html=True)
        render_coaching(coaching)
    st.markdown("</div></div>", unsafe_allow_html=True)


if "view" not in st.session_state:
    st.session_state.view = "splash"
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

css()

if st.session_state.view == "splash":
    start_screen()
elif st.session_state.view == "home":
    home_screen()
elif st.session_state.view == "coming_soon":
    coming_soon_screen()
elif st.session_state.view == "loading_dashboard":
    loading_screen()
else:
    render_app_nav(show_back=True)
    st.markdown('<div class="dev-info">', unsafe_allow_html=True)
    with st.expander("Development Information"):
        render_connection_information()
        render_external_demo_access()
    st.markdown('</div>', unsafe_allow_html=True)
    render_subject_information()
    dashboard()
