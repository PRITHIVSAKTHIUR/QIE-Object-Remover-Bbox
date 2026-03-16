import gradio as gr
import numpy as np
import random
import torch
import spaces
import base64
import json
from io import BytesIO
from PIL import Image, ImageDraw
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

MAX_SEED = np.iinfo(np.int32).max

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V4",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)
try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

ADAPTER_SPECS = {
"Object-Remover": {
        "repo": "prithivMLmods/QIE-2509-Object-Remover-Bbox-v3",
        "weights": "QIE-2509-Object-Remover-Bbox-v3-10000.safetensors",
        "adapter_name": "object-remover",
    },
}
loaded = False

DEFAULT_PROMPT = "Remove the red highlighted object from the scene"


def b64_to_pil(b64_str):
    if not b64_str or not b64_str.startswith("data:image"):
        return None
    try:
        _, data = b64_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def burn_boxes_onto_image(pil_image, boxes_json_str):
    if not pil_image:
        return pil_image
    try:
        boxes = json.loads(boxes_json_str) if boxes_json_str and boxes_json_str.strip() else []
    except Exception:
        boxes = []
    if not boxes:
        return pil_image

    img = pil_image.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    bw = max(3, w // 250)

    for b in boxes:
        x1 = int(b["x1"] * w)
        y1 = int(b["y1"] * h)
        x2 = int(b["x2"] * w)
        y2 = int(b["y2"] * h)
        lx, rx = min(x1, x2), max(x1, x2)
        ty, by_ = min(y1, y2), max(y1, y2)
        draw.rectangle([lx, ty, rx, by_], outline=(255, 0, 0), width=bw)

    return img


@spaces.GPU
def infer_object_removal(
    b64_str,
    boxes_json,
    prompt,
    seed=0,
    randomize_seed=True,
    guidance_scale=1.0,
    num_inference_steps=4,
    height=1024,
    width=1024,
):
    global loaded
    progress = gr.Progress(track_tqdm=True)

    if not loaded:
        pipe.load_lora_weights(
            ADAPTER_SPECS["Object-Remover"]["repo"],
            weight_name=ADAPTER_SPECS["Object-Remover"]["weights"],
            adapter_name=ADAPTER_SPECS["Object-Remover"]["adapter_name"],
        )
        pipe.set_adapters(
            [ADAPTER_SPECS["Object-Remover"]["adapter_name"]], adapter_weights=[1.0]
        )
        loaded = True

    if not prompt or prompt.strip() == "":
        prompt = DEFAULT_PROMPT

    source_image = b64_to_pil(b64_str)
    if source_image is None:
        raise gr.Error("Please upload an image first using the canvas area.")

    try:
        boxes = json.loads(boxes_json) if boxes_json and boxes_json.strip() else []
    except Exception:
        boxes = []

    if not boxes:
        raise gr.Error("Please draw at least one bounding box on the image.")

    progress(0.3, desc="Burning red boxes onto image...")
    marked = burn_boxes_onto_image(source_image, boxes_json)

    progress(0.5, desc="Running object removal inference...")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        image=[marked],
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed, marked


def update_dimensions_on_upload(b64_str):
    image = b64_to_pil(b64_str)
    if image is None:
        return 1024, 1024
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    return new_width, new_height


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*{box-sizing:border-box;margin:0;padding:0}

body,.gradio-container{
    background:#0f0f13!important;
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;
    color:#e4e4e7!important;
    min-height:100vh;
}
.dark body,.dark .gradio-container{
    background:#0f0f13!important;
    color:#e4e4e7!important;
}
footer{display:none!important}

.hidden-input{
    display:none!important;
    height:0!important;
    overflow:hidden!important;
    margin:0!important;
    padding:0!important;
}

/* ── Main Container ── */
.app-shell{
    background:#18181b;
    border:1px solid #27272a;
    border-radius:16px;
    margin:12px auto;
    max-width:1400px;
    overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),
               0 0 0 1px rgba(255,255,255,.03);
}

/* ── Header Bar ── */
.app-header{
    background:linear-gradient(135deg,#18181b 0%,#1e1e24 100%);
    border-bottom:1px solid #27272a;
    padding:14px 24px;
    display:flex;
    align-items:center;
    justify-content:space-between;
}
.app-header-left{
    display:flex;
    align-items:center;
    gap:12px;
}
.app-logo{
    width:36px;height:36px;
    background:linear-gradient(135deg,#6366f1,#8b5cf6,#a78bfa);
    border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:18px;font-weight:800;color:#fff;
    box-shadow:0 4px 12px rgba(99,102,241,.35);
}
.app-title{
    font-size:18px;font-weight:700;
    background:linear-gradient(135deg,#e4e4e7,#a1a1aa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    letter-spacing:-.3px;
}
.app-badge{
    font-size:11px;
    font-weight:600;
    padding:3px 10px;
    border-radius:20px;
    background:rgba(99,102,241,.15);
    color:#818cf8;
    border:1px solid rgba(99,102,241,.25);
    letter-spacing:.3px;
}

/* ── Toolbar ── */
.app-toolbar{
    background:#18181b;
    border-bottom:1px solid #27272a;
    padding:8px 16px;
    display:flex;
    gap:4px;
    align-items:center;
    flex-wrap:wrap;
}
.tb-sep{
    width:1px;height:28px;
    background:#27272a;
    margin:0 8px;
}
.modern-tb-btn{
    display:inline-flex;align-items:center;justify-content:center;
    gap:6px;
    min-width:32px;height:34px;
    background:transparent;
    border:1px solid transparent;
    border-radius:8px;
    cursor:pointer;
    font-size:13px;
    font-weight:600;
    padding:0 12px;
    font-family:'Inter',sans-serif;
    color:#ffffff!important;
    transition:all .15s ease;
}
.modern-tb-btn:hover{
    background:rgba(99,102,241,.15);
    color:#ffffff!important;
    border-color:rgba(99,102,241,.3);
}
.modern-tb-btn:active,.modern-tb-btn.active{
    background:rgba(99,102,241,.25);
    color:#ffffff!important;
    border-color:rgba(99,102,241,.45);
}
.modern-tb-btn .tb-icon{
    font-size:15px;
    line-height:1;
    color:#ffffff!important;
}
.modern-tb-btn .tb-label{
    font-size:13px;
    color:#ffffff!important;
    font-weight:600;
}

/* ── Main Layout ── */
.app-main-row{
    display:flex;
    gap:0;
    flex:1;
    overflow:hidden;
}
.app-main-left{
    flex:1;
    display:flex;
    flex-direction:column;
    min-width:0;
    border-right:1px solid #27272a;
}
.app-main-right{
    width:420px;
    display:flex;
    flex-direction:column;
    flex-shrink:0;
    background:#18181b;
}

/* ── Canvas Area ── */
#bbox-draw-wrap{
    position:relative;
    background:#09090b;
    margin:0;
    min-height:440px;
    overflow:hidden;
    cursor:crosshair;
}
#bbox-draw-canvas{display:block;margin:0 auto}
#bbox-status{
    position:absolute;top:12px;left:12px;
    background:rgba(99,102,241,.9);
    color:#fff;
    padding:4px 12px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    font-weight:500;
    border-radius:6px;
    z-index:10;display:none;
    pointer-events:none;
    backdrop-filter:blur(8px);
}
#bbox-count{
    position:absolute;top:12px;right:12px;
    background:rgba(24,24,27,.9);
    color:#a78bfa;
    padding:4px 12px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    font-weight:600;
    border-radius:6px;
    border:1px solid rgba(99,102,241,.3);
    z-index:10;display:none;
    backdrop-filter:blur(8px);
}

/* ── Upload Prompt (icon only) ── */
.upload-prompt-modern{
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    z-index:20;
}
.upload-click-area{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    padding:36px 44px;
    border:2px dashed #3f3f46;
    border-radius:16px;
    background:rgba(99,102,241,.03);
    transition:all .2s ease;
}
.upload-click-area:hover{
    background:rgba(99,102,241,.08);
    border-color:#6366f1;
    transform:scale(1.03);
}
.upload-click-area:active{
    background:rgba(99,102,241,.12);
    transform:scale(.98);
}
.upload-click-area svg{
    width:80px;height:80px;
}

/* ── Hint Bar ── */
.hint-bar{
    background:rgba(99,102,241,.06);
    border-top:1px solid #27272a;
    border-bottom:1px solid #27272a;
    padding:10px 20px;
    font-size:13px;
    color:#a1a1aa;
    line-height:1.7;
}
.hint-bar b{color:#c7d2fe;font-weight:600}
.hint-bar kbd{
    display:inline-block;
    padding:1px 6px;
    background:#27272a;
    border:1px solid #3f3f46;
    border-radius:4px;
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    color:#a1a1aa;
}

/* ── JSON Panel ── */
.json-panel{
    background:#18181b;
    border-top:1px solid #27272a;
    display:flex;
    flex-direction:column;
    height:160px;
    max-height:160px;
    min-height:160px;
}
.json-panel-title{
    padding:8px 16px;
    font-size:12px;
    font-weight:600;
    color:#71717a;
    text-transform:uppercase;
    letter-spacing:.8px;
    border-bottom:1px solid #27272a;
    display:flex;
    align-items:center;
    gap:8px;
    flex-shrink:0;
}
.json-panel-title::before{
    content:'{ }';
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    color:#6366f1;
    background:rgba(99,102,241,.12);
    padding:2px 6px;
    border-radius:4px;
}
.json-panel-content{
    background:#09090b;
    margin:0;
    padding:12px 16px;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#a1a1aa;
    flex:1;
    overflow-y:auto;
    overflow-x:hidden;
    word-break:break-all;
    white-space:pre-wrap;
    line-height:1.6;
}
.json-panel-content::-webkit-scrollbar{width:8px}
.json-panel-content::-webkit-scrollbar-track{background:#09090b}
.json-panel-content::-webkit-scrollbar-thumb{
    background:#27272a;
    border-radius:4px;
}
.json-panel-content::-webkit-scrollbar-thumb:hover{background:#3f3f46}

/* ── Right Panel Cards ── */
.panel-card{
    border-bottom:1px solid #27272a;
}
.panel-card-title{
    padding:12px 20px;
    font-size:12px;
    font-weight:600;
    color:#71717a;
    text-transform:uppercase;
    letter-spacing:.8px;
    border-bottom:1px solid rgba(39,39,42,.6);
}
.panel-card-body{
    padding:16px 20px;
    display:flex;
    flex-direction:column;
    gap:8px;
}

.modern-label{
    font-size:13px;font-weight:500;color:#a1a1aa;margin-bottom:4px;display:block;
}
.modern-textarea{
    width:100%;
    background:#09090b;
    border:1px solid #27272a;
    border-radius:8px;
    padding:10px 14px;
    font-family:'Inter',sans-serif;
    font-size:14px;
    color:#e4e4e7;
    resize:vertical;
    outline:none;
    min-height:42px;
    transition:border-color .2s;
}
.modern-textarea:focus{
    border-color:#6366f1;
    box-shadow:0 0 0 3px rgba(99,102,241,.15);
}
.modern-textarea::placeholder{color:#3f3f46}

/* ── Primary Button ── */
.btn-run{
    display:flex;align-items:center;justify-content:center;gap:8px;
    width:100%;
    background:linear-gradient(135deg,#6366f1,#7c3aed);
    border:none;
    border-radius:10px;
    padding:12px 24px;
    cursor:pointer;
    font-size:15px;
    font-weight:600;
    font-family:'Inter',sans-serif;
    color:#fff;
    transition:all .2s ease;
    box-shadow:0 4px 16px rgba(99,102,241,.3),
               inset 0 1px 0 rgba(255,255,255,.1);
    letter-spacing:-.2px;
}
.btn-run:hover{
    background:linear-gradient(135deg,#7c7cf5,#8b5cf6);
    box-shadow:0 6px 24px rgba(99,102,241,.45),
               inset 0 1px 0 rgba(255,255,255,.15);
    transform:translateY(-1px);
}
.btn-run:active{
    transform:translateY(0);
    box-shadow:0 2px 8px rgba(99,102,241,.3);
}
.btn-run svg{width:18px;height:18px;fill:#fff}

/* ── Output Frames ── */
.output-frame{
    border-bottom:1px solid #27272a;
    display:flex;
    flex-direction:column;
    position:relative;
}
.output-frame .out-title{
    padding:10px 20px;
    font-size:13px;
    font-weight:700;
    color:#ffffff!important;
    text-transform:uppercase;
    letter-spacing:.8px;
    border-bottom:1px solid rgba(39,39,42,.6);
    display:flex;
    align-items:center;
    justify-content:space-between;
}
.output-frame .out-title span{
    color:#ffffff!important;
}
.output-frame .out-body{
    flex:1;
    background:#09090b;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:hidden;
    min-height:180px;
    position:relative;
}
.output-frame .out-body img{
    max-width:100%;max-height:460px;
    image-rendering:auto;
}
.output-frame .out-placeholder{
    color:#3f3f46;
    font-size:13px;
    text-align:center;
    padding:20px;
}

.out-download-btn{
    display:none;
    align-items:center;
    justify-content:center;
    background:rgba(99,102,241,.1);
    border:1px solid rgba(99,102,241,.2);
    border-radius:6px;
    cursor:pointer;
    padding:3px 10px;
    font-size:11px;
    font-weight:500;
    color:#c7d2fe!important;
    gap:4px;
    height:24px;
    transition:all .15s;
}
.out-download-btn:hover{
    background:rgba(99,102,241,.2);
    border-color:rgba(99,102,241,.35);
    color:#ffffff!important;
}
.out-download-btn.visible{display:inline-flex}
.out-download-btn svg{width:12px;height:12px;fill:#c7d2fe}

/* ── Loader ── */
.modern-loader{
    display:none;
    position:absolute;
    top:0;left:0;right:0;bottom:0;
    background:rgba(9,9,11,.92);
    z-index:15;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    gap:16px;
    backdrop-filter:blur(4px);
}
.modern-loader.active{display:flex}
.modern-loader .loader-spinner{
    width:36px;height:36px;
    border:3px solid #27272a;
    border-top-color:#6366f1;
    border-radius:50%;
    animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.modern-loader .loader-text{
    font-size:13px;
    color:#a1a1aa;
    font-weight:500;
}
.loader-bar-track{
    width:200px;height:4px;
    background:#27272a;
    border-radius:2px;
    overflow:hidden;
}
.loader-bar-fill{
    height:100%;
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#6366f1);
    background-size:200% 100%;
    animation:shimmer 1.5s ease-in-out infinite;
    border-radius:2px;
}
@keyframes shimmer{
    0%{background-position:200% 0}
    100%{background-position:-200% 0}
}

/* ── Settings ── */
.settings-group{
    border:1px solid #27272a;
    border-radius:10px;
    margin:12px 16px;
    padding:0;
    overflow:hidden;
}
.settings-group-title{
    font-size:12px;
    font-weight:600;
    color:#71717a;
    text-transform:uppercase;
    letter-spacing:.8px;
    padding:10px 16px;
    border-bottom:1px solid #27272a;
    background:rgba(24,24,27,.5);
}
.settings-group-body{
    padding:14px 16px;
    display:flex;
    flex-direction:column;
    gap:12px;
}
.slider-row{
    display:flex;
    align-items:center;
    gap:10px;
    min-height:28px;
}
.slider-row label,.slider-row .dim-label{
    font-size:13px;
    font-weight:500;
    color:#a1a1aa;
    min-width:72px;
    flex-shrink:0;
}
.slider-row input[type="range"]{
    flex:1;
    -webkit-appearance:none;
    appearance:none;
    height:6px;
    background:#27272a;
    border-radius:3px;
    outline:none;
    min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb{
    -webkit-appearance:none;
    appearance:none;
    width:16px;height:16px;
    background:linear-gradient(135deg,#6366f1,#7c3aed);
    border-radius:50%;
    cursor:pointer;
    box-shadow:0 2px 6px rgba(99,102,241,.4);
    transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{
    transform:scale(1.2);
}
.slider-row input[type="range"]::-moz-range-thumb{
    width:16px;height:16px;
    background:linear-gradient(135deg,#6366f1,#7c3aed);
    border-radius:50%;
    cursor:pointer;
    border:none;
    box-shadow:0 2px 6px rgba(99,102,241,.4);
}
.slider-row .slider-val{
    min-width:52px;text-align:right;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    font-weight:500;
    padding:3px 8px;
    background:#09090b;
    border:1px solid #27272a;
    border-radius:6px;
    color:#a1a1aa;
    flex-shrink:0;
}
.checkbox-row{
    display:flex;align-items:center;gap:8px;
    font-size:13px;cursor:default;
    color:#a1a1aa;
}
.checkbox-row input[type="checkbox"]{
    accent-color:#6366f1;
    width:16px;height:16px;
    cursor:pointer;
}
.checkbox-row label{
    color:#a1a1aa;font-size:13px;cursor:pointer;
}

/* ── Status Bar ── */
.app-statusbar{
    background:#18181b;
    border-top:1px solid #27272a;
    padding:6px 20px;
    display:flex;
    gap:12px;
    height:34px;
    align-items:center;
    font-size:12px;
}
.app-statusbar .sb-section{
    padding:0 12px;
    flex:1;
    display:flex;align-items:center;
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#52525b;
    overflow:hidden;
    white-space:nowrap;
}
.app-statusbar .sb-section.sb-fixed{
    flex:0 0 auto;
    min-width:90px;
    text-align:center;
    justify-content:center;
    padding:3px 12px;
    background:rgba(99,102,241,.08);
    border-radius:6px;
    color:#818cf8;
    font-weight:500;
}

#gradio-run-btn{
    position:absolute;
    left:-9999px;
    top:-9999px;
    width:1px;
    height:1px;
    opacity:0.01;
    pointer-events:none;
    overflow:hidden;
}

#bbox-debug-count{
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#52525b;
}

/* ── Global scrollbar ── */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#09090b}
::-webkit-scrollbar-thumb{background:#27272a;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#3f3f46}

/* ── Dark mode force-overrides ── */
.dark .app-shell{background:#18181b}
.dark .upload-prompt-modern{background:transparent}
.dark .panel-card{background:#18181b}
.dark .settings-group{background:#18181b}
.dark .modern-tb-btn{color:#ffffff!important}
.dark .modern-tb-btn .tb-icon{color:#ffffff!important}
.dark .modern-tb-btn .tb-label{color:#ffffff!important}
.dark .modern-tb-btn:hover{color:#ffffff!important}
.dark .modern-tb-btn:active,.dark .modern-tb-btn.active{color:#ffffff!important}
.dark .output-frame .out-title{color:#ffffff!important}
.dark .output-frame .out-title span{color:#ffffff!important}
.dark .out-download-btn{color:#c7d2fe!important}
.dark .out-download-btn:hover{color:#ffffff!important}

@media(max-width:840px){
    .app-main-row{flex-direction:column}
    .app-main-right{width:100%}
    .app-main-left{border-right:none;border-bottom:1px solid #27272a}
}
"""

bbox_drawer_js = r"""
() => {
function initCanvasBbox() {
    if (window.__bboxInitDone) return;

    const canvas     = document.getElementById('bbox-draw-canvas');
    const wrap       = document.getElementById('bbox-draw-wrap');
    const status     = document.getElementById('bbox-status');
    const badge      = document.getElementById('bbox-count');
    const debugCount = document.getElementById('bbox-debug-count');
    const jsonDisplay = document.getElementById('bbox-json-content');

    const btnDraw    = document.getElementById('tb-draw');
    const btnSelect  = document.getElementById('tb-select');
    const btnReset   = document.getElementById('tb-reset');
    const btnDel     = document.getElementById('tb-del');
    const btnUndo    = document.getElementById('tb-undo');
    const btnClear   = document.getElementById('tb-clear');
    const btnChange  = document.getElementById('tb-change-img');

    const uploadPrompt    = document.getElementById('upload-prompt');
    const uploadClickArea = document.getElementById('upload-click-area');
    const fileInput       = document.getElementById('custom-file-input');

    if (!canvas || !wrap || !debugCount || !btnDraw || !fileInput) {
        setTimeout(initCanvasBbox, 250);
        return;
    }

    window.__bboxInitDone = true;
    const ctx = canvas.getContext('2d');

    let boxes = [];
    window.__bboxBoxes = boxes;

    let baseImg = null;
    let dispW = 512, dispH = 400;
    let selectedIdx = -1;
    let mode = 'draw';

    let dragging  = false;
    let dragType  = null;
    let dragStart = {x:0, y:0};
    let dragOrig  = null;
    const HANDLE  = 6;
    const RED_STROKE = 'rgba(239,68,68,0.95)';
    const RED_STROKE_WIDTH = 2;
    const SEL_STROKE = 'rgba(99,102,241,0.95)';

    function n2px(b) { return {x1:b.x1*dispW, y1:b.y1*dispH, x2:b.x2*dispW, y2:b.y2*dispH}; }
    function px2n(x1,y1,x2,y2) {
        return {
            x1: Math.min(x1,x2)/dispW, y1: Math.min(y1,y2)/dispH,
            x2: Math.max(x1,x2)/dispW, y2: Math.max(y1,y2)/dispH
        };
    }
    function clamp01(v){return Math.max(0,Math.min(1,v));}
    function fitSize(nw, nh) {
        const mw = wrap.clientWidth || 512, mh = 500;
        const r = Math.min(mw/nw, mh/nh, 1);
        dispW = Math.round(nw*r); dispH = Math.round(nh*r);
        canvas.width  = dispW; canvas.height = dispH;
        canvas.style.width  = dispW+'px';
        canvas.style.height = dispH+'px';
    }
    function canvasXY(e) {
        const r  = canvas.getBoundingClientRect();
        const cx = e.touches ? e.touches[0].clientX : e.clientX;
        const cy = e.touches ? e.touches[0].clientY : e.clientY;
        return {x: Math.max(0,Math.min(dispW, cx-r.left)),
                y: Math.max(0,Math.min(dispH, cy-r.top))};
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return;
        const allInputs = container.querySelectorAll('input, textarea');
        allInputs.forEach(el => {
            if (el.type === 'file' || el.type === 'range' || el.type === 'checkbox') return;
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, value);
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }

    function formatJsonPretty(boxes) {
        if (!boxes || boxes.length === 0) return '[\n  // No bounding boxes defined\n]';
        let lines = '[\n';
        boxes.forEach((b, i) => {
            lines += '  {\n';
            lines += '    "x1": ' + b.x1.toFixed(4) + ',\n';
            lines += '    "y1": ' + b.y1.toFixed(4) + ',\n';
            lines += '    "x2": ' + b.x2.toFixed(4) + ',\n';
            lines += '    "y2": ' + b.y2.toFixed(4) + '\n';
            lines += '  }';
            if (i < boxes.length - 1) lines += ',';
            lines += '\n';
        });
        lines += ']';
        return lines;
    }

    function syncToGradio() {
        window.__bboxBoxes = boxes;
        const jsonStr = JSON.stringify(boxes);
        if (debugCount) {
            debugCount.textContent = boxes.length > 0
                ? boxes.length + ' box' + (boxes.length > 1 ? 'es' : '') + ' drawn'
                : 'No boxes drawn';
        }
        if (jsonDisplay) {
            jsonDisplay.textContent = formatJsonPretty(boxes);
            jsonDisplay.scrollTop = jsonDisplay.scrollHeight;
        }
        setGradioValue('boxes-json-input', jsonStr);
    }

    function syncImageToGradio(dataUrl) {
        setGradioValue('hidden-image-b64', dataUrl);
    }

    function syncPromptToGradio() {
        const promptInput = document.getElementById('custom-prompt-input');
        if (promptInput) {
            setGradioValue('prompt-gradio-input', promptInput.value);
        }
    }

    function resetCanvas() {
        baseImg = null;
        boxes.length = 0;
        window.__bboxBoxes = boxes;
        selectedIdx = -1;
        dragging = false;
        dragType = null;
        dragOrig = null;
        fitSize(512, 400);
        syncToGradio();
        syncImageToGradio('');
        redraw();
        hideStatus();
        uploadPrompt.style.display = '';
        showStatus('Image removed');
        setTimeout(hideStatus, 1500);
    }

    function redraw(tempRect) {
        ctx.clearRect(0,0,dispW,dispH);
        if (!baseImg) {
            ctx.fillStyle='#09090b'; ctx.fillRect(0,0,dispW,dispH);
            updateBadge(); return;
        }
        ctx.drawImage(baseImg, 0, 0, dispW, dispH);

        boxes.forEach((b,i) => {
            const p = n2px(b);
            const lx=p.x1, ty=p.y1, w=p.x2-p.x1, h=p.y2-p.y1;
            if (i === selectedIdx) {
                ctx.strokeStyle = SEL_STROKE;
                ctx.lineWidth = RED_STROKE_WIDTH + 1;
                ctx.setLineDash([4,3]);
            } else {
                ctx.strokeStyle = RED_STROKE;
                ctx.lineWidth = RED_STROKE_WIDTH;
                ctx.setLineDash([]);
            }
            ctx.strokeRect(lx, ty, w, h);
            ctx.setLineDash([]);

            ctx.fillStyle = i===selectedIdx ? '#6366f1' : '#ef4444';
            ctx.font = 'bold 11px Inter,system-ui,sans-serif';
            ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const label = '#'+(i+1);
            const tw = ctx.measureText(label).width;
            const rx = lx, ry = ty - 18;
            const rw = tw + 10, rh = 18;
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(rx, ry, rw, rh, 3);
            } else {
                ctx.rect(rx, ry, rw, rh);
            }
            ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.fillText(label, lx+5, ty-15);

            if (i === selectedIdx) drawHandles(p);
        });

        if (tempRect) {
            const rx = Math.min(tempRect.x1,tempRect.x2);
            const ry = Math.min(tempRect.y1,tempRect.y2);
            const rw = Math.abs(tempRect.x2-tempRect.x1);
            const rh = Math.abs(tempRect.y2-tempRect.y1);
            ctx.strokeStyle = RED_STROKE;
            ctx.lineWidth = RED_STROKE_WIDTH;
            ctx.setLineDash([4,3]);
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.setLineDash([]);
        }
        updateBadge();
    }

    function drawHandles(p) {
        const pts = handlePoints(p);
        for (const k in pts) {
            const h = pts[k];
            ctx.fillStyle = '#6366f1';
            ctx.beginPath();
            ctx.arc(h.x, h.y, HANDLE, 0, Math.PI*2);
            ctx.fill();
            ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(h.x, h.y, HANDLE, 0, Math.PI*2);
            ctx.stroke();
        }
    }

    function handlePoints(p) {
        const mx = (p.x1+p.x2)/2, my = (p.y1+p.y2)/2;
        return {
            tl:{x:p.x1,y:p.y1}, tc:{x:mx,y:p.y1}, tr:{x:p.x2,y:p.y1},
            ml:{x:p.x1,y:my},                       mr:{x:p.x2,y:my},
            bl:{x:p.x1,y:p.y2}, bc:{x:mx,y:p.y2}, br:{x:p.x2,y:p.y2}
        };
    }

    function hitHandle(px, py, boxIdx) {
        if (boxIdx < 0) return null;
        const p = n2px(boxes[boxIdx]);
        const pts = handlePoints(p);
        for (const k in pts) {
            if (Math.abs(px-pts[k].x) <= HANDLE+2 && Math.abs(py-pts[k].y) <= HANDLE+2) return k;
        }
        return null;
    }

    function hitBox(px, py) {
        for (let i = boxes.length-1; i >= 0; i--) {
            const p = n2px(boxes[i]);
            if (px >= p.x1 && px <= p.x2 && py >= p.y1 && py <= p.y2) return i;
        }
        return -1;
    }

    function updateBadge() {
        if (boxes.length > 0) {
            badge.style.display = 'block';
            badge.textContent = boxes.length + ' box' + (boxes.length>1?'es':'');
        } else {
            badge.style.display = 'none';
        }
    }

    function setMode(m) {
        mode = m;
        btnDraw.classList.toggle('active', m==='draw');
        btnSelect.classList.toggle('active', m==='select');
        canvas.style.cursor = m==='draw' ? 'crosshair' : 'default';
        if (m==='draw') selectedIdx = -1;
        redraw();
    }

    function showStatus(txt) {
        status.textContent = txt; status.style.display = 'block';
    }
    function hideStatus() { status.style.display = 'none'; }

    function onDown(e) {
        if (!baseImg) return;
        e.preventDefault();
        const {x, y} = canvasXY(e);
        if (mode === 'draw') {
            dragging = true; dragType = 'new';
            dragStart = {x, y};
            selectedIdx = -1;
        } else {
            if (selectedIdx >= 0) {
                const h = hitHandle(x, y, selectedIdx);
                if (h) {
                    dragging = true; dragType = h;
                    dragStart = {x, y};
                    dragOrig = {...boxes[selectedIdx]};
                    showStatus('Resizing #'+(selectedIdx+1));
                    return;
                }
            }
            const hi = hitBox(x, y);
            if (hi >= 0) {
                selectedIdx = hi;
                const h2 = hitHandle(x, y, selectedIdx);
                if (h2) {
                    dragging = true; dragType = h2;
                    dragStart = {x, y};
                    dragOrig = {...boxes[selectedIdx]};
                    showStatus('Resizing #'+(selectedIdx+1));
                    redraw(); return;
                }
                dragging = true; dragType = 'move';
                dragStart = {x, y};
                dragOrig = {...boxes[selectedIdx]};
                showStatus('Moving #'+(selectedIdx+1));
            } else {
                selectedIdx = -1;
                hideStatus();
            }
            redraw();
        }
    }

    function onMove(e) {
        if (!baseImg) return;
        e.preventDefault();
        const {x, y} = canvasXY(e);
        if (!dragging) {
            if (mode === 'select') {
                if (selectedIdx >= 0 && hitHandle(x,y,selectedIdx)) {
                    const h = hitHandle(x,y,selectedIdx);
                    const curs = {tl:'nwse-resize',tr:'nesw-resize',bl:'nesw-resize',br:'nwse-resize',
                                  tc:'ns-resize',bc:'ns-resize',ml:'ew-resize',mr:'ew-resize'};
                    canvas.style.cursor = curs[h] || 'move';
                } else if (hitBox(x,y) >= 0) {
                    canvas.style.cursor = 'move';
                } else {
                    canvas.style.cursor = 'default';
                }
            }
            return;
        }
        if (dragType === 'new') {
            redraw({x1:dragStart.x, y1:dragStart.y, x2:x, y2:y});
            showStatus(Math.abs(x-dragStart.x).toFixed(0)+'\u00d7'+Math.abs(y-dragStart.y).toFixed(0)+' px');
            return;
        }
        const dx = (x - dragStart.x) / dispW;
        const dy = (y - dragStart.y) / dispH;
        const b  = boxes[selectedIdx];
        const o  = dragOrig;
        if (dragType === 'move') {
            const bw = o.x2-o.x1, bh = o.y2-o.y1;
            let nx1 = o.x1+dx, ny1 = o.y1+dy;
            nx1 = clamp01(nx1); ny1 = clamp01(ny1);
            if (nx1+bw > 1) nx1 = 1-bw;
            if (ny1+bh > 1) ny1 = 1-bh;
            b.x1=nx1; b.y1=ny1; b.x2=nx1+bw; b.y2=ny1+bh;
        } else {
            const t = dragType;
            if (t.includes('l')) b.x1 = clamp01(o.x1 + dx);
            if (t.includes('r')) b.x2 = clamp01(o.x2 + dx);
            if (t.includes('t')) b.y1 = clamp01(o.y1 + dy);
            if (t.includes('b')) b.y2 = clamp01(o.y2 + dy);
            if (Math.abs(b.x2-b.x1) < 0.01) { b.x1=o.x1; b.x2=o.x2; }
            if (Math.abs(b.y2-b.y1) < 0.01) { b.y1=o.y1; b.y2=o.y2; }
            if (b.x1 > b.x2) { const t2=b.x1; b.x1=b.x2; b.x2=t2; }
            if (b.y1 > b.y2) { const t2=b.y1; b.y1=b.y2; b.y2=t2; }
        }
        redraw();
    }

    function onUp(e) {
        if (!dragging) return;
        if (e) e.preventDefault();
        dragging = false;
        if (dragType === 'new') {
            const pt = e ? canvasXY(e) : {x:dragStart.x, y:dragStart.y};
            if (Math.abs(pt.x-dragStart.x) > 4 && Math.abs(pt.y-dragStart.y) > 4) {
                const nb = px2n(dragStart.x, dragStart.y, pt.x, pt.y);
                boxes.push(nb);
                window.__bboxBoxes = boxes;
                selectedIdx = boxes.length - 1;
                showStatus('Box #'+boxes.length+' created');
            } else { hideStatus(); }
        } else {
            showStatus('Box #'+(selectedIdx+1)+' updated');
        }
        dragType = null; dragOrig = null;
        syncToGradio();
        redraw();
    }

    canvas.addEventListener('mousedown',  onDown);
    canvas.addEventListener('mousemove',  onMove);
    canvas.addEventListener('mouseup',    onUp);
    canvas.addEventListener('mouseleave', (e)=>{if(dragging)onUp(e);});
    canvas.addEventListener('touchstart', onDown, {passive:false});
    canvas.addEventListener('touchmove',  onMove, {passive:false});
    canvas.addEventListener('touchend',   onUp,   {passive:false});
    canvas.addEventListener('touchcancel',(e)=>{e.preventDefault();dragging=false;redraw();},{passive:false});

    function processFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const dataUrl = event.target.result;
            const img = new window.Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                baseImg = img;
                boxes.length = 0;
                window.__bboxBoxes = boxes;
                selectedIdx = -1;
                fitSize(img.naturalWidth, img.naturalHeight);
                syncToGradio(); redraw(); hideStatus();
                uploadPrompt.style.display = 'none';
                syncImageToGradio(dataUrl);
            };
            img.src = dataUrl;
        };
        reader.readAsDataURL(file);
    }

    function openFilePicker() {
        fileInput.click();
    }

    uploadClickArea.addEventListener('click', openFilePicker);
    btnChange.addEventListener('click', openFilePicker);

    fileInput.addEventListener('change', (e) => {
        processFile(e.target.files[0]);
        e.target.value = '';
    });

    wrap.addEventListener('dragover', (e) => {
        e.preventDefault();
        wrap.style.outline = '2px solid #6366f1';
        wrap.style.outlineOffset = '-2px';
    });
    wrap.addEventListener('dragleave', (e) => {
        e.preventDefault();
        wrap.style.outline = '';
    });
    wrap.addEventListener('drop', (e) => {
        e.preventDefault();
        wrap.style.outline = '';
        if (e.dataTransfer.files.length) {
            processFile(e.dataTransfer.files[0]);
        }
    });

    btnDraw.addEventListener('click',   ()=>setMode('draw'));
    btnSelect.addEventListener('click', ()=>setMode('select'));

    btnReset.addEventListener('click', () => {
        resetCanvas();
    });

    btnDel.addEventListener('click', () => {
        if (selectedIdx >= 0 && selectedIdx < boxes.length) {
            const removed = selectedIdx + 1;
            boxes.splice(selectedIdx, 1);
            window.__bboxBoxes = boxes;
            selectedIdx = -1;
            syncToGradio(); redraw();
            showStatus('Box #'+removed+' deleted');
        } else {
            showStatus('Select a box first');
        }
    });

    btnUndo.addEventListener('click', () => {
        if (boxes.length > 0) {
            boxes.pop();
            window.__bboxBoxes = boxes;
            selectedIdx = -1;
            syncToGradio(); redraw();
            showStatus('Last box removed');
        }
    });

    btnClear.addEventListener('click', () => {
        boxes.length = 0;
        window.__bboxBoxes = boxes;
        selectedIdx = -1;
        syncToGradio(); redraw(); hideStatus();
    });

    const promptInput = document.getElementById('custom-prompt-input');
    if (promptInput) {
        promptInput.addEventListener('input', () => {
            syncPromptToGradio();
        });
    }

    function syncSlider(customId, gradioId) {
        const slider = document.getElementById(customId);
        const valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const container = document.getElementById(gradioId);
            if (!container) return;
            const targets = [
                ...container.querySelectorAll('input[type="range"]'),
                ...container.querySelectorAll('input[type="number"]')
            ];
            targets.forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }
    syncSlider('custom-seed', 'gradio-seed');
    syncSlider('custom-guidance', 'gradio-guidance');
    syncSlider('custom-steps', 'gradio-steps');
    syncSlider('custom-height', 'gradio-height');
    syncSlider('custom-width', 'gradio-width');

    const randCheck = document.getElementById('custom-randomize');
    if (randCheck) {
        randCheck.addEventListener('change', () => {
            const container = document.getElementById('gradio-randomize');
            if (!container) return;
            const cb = container.querySelector('input[type="checkbox"]');
            if (cb && cb.checked !== randCheck.checked) {
                cb.click();
            }
        });
    }

    function showLoaders() {
        const l1 = document.getElementById('output-loader');
        const l2 = document.getElementById('preview-loader');
        if (l1) l1.classList.add('active');
        if (l2) l2.classList.add('active');
        const sb = document.querySelector('.app-statusbar .sb-fixed');
        if (sb) sb.textContent = 'Processing...';
    }

    function hideLoaders() {
        const l1 = document.getElementById('output-loader');
        const l2 = document.getElementById('preview-loader');
        if (l1) l1.classList.remove('active');
        if (l2) l2.classList.remove('active');
        const sb = document.querySelector('.app-statusbar .sb-fixed');
        if (sb) sb.textContent = 'Done';
    }

    window.__showLoaders = showLoaders;
    window.__hideLoaders = hideLoaders;

    window.__clickGradioRunBtn = function() {
        syncPromptToGradio();
        syncToGradio();
        showLoaders();

        setTimeout(() => {
            const gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) return;
            const btn = gradioBtn.querySelector('button');
            if (btn) {
                btn.click();
            } else {
                gradioBtn.click();
            }
        }, 200);
    };

    const customRunBtn = document.getElementById('custom-run-btn');
    if (customRunBtn) {
        customRunBtn.addEventListener('click', () => {
            window.__clickGradioRunBtn();
        });
    }

    new ResizeObserver(() => {
        if (baseImg) { fitSize(baseImg.naturalWidth, baseImg.naturalHeight); redraw(); }
    }).observe(wrap);

    setMode('draw');
    fitSize(512,400); redraw();
    syncToGradio();
}

initCanvasBbox();
}
"""

wire_outputs_js = r"""
() => {
function downloadImage(imgSrc, filename) {
    const a = document.createElement('a');
    a.href = imgSrc;
    a.download = filename || 'image.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function watchOutputs() {
    const resultContainer = document.getElementById('gradio-result');
    const previewContainer = document.getElementById('gradio-preview');
    const outBody = document.getElementById('output-image-container');
    const prevBody = document.getElementById('preview-image-container');
    const outPh = document.getElementById('output-placeholder');
    const prevPh = document.getElementById('preview-placeholder');
    const dlBtnOut = document.getElementById('dl-btn-output');
    const dlBtnPrev = document.getElementById('dl-btn-preview');

    if (!resultContainer || !previewContainer || !outBody || !prevBody) {
        setTimeout(watchOutputs, 500);
        return;
    }

    if (dlBtnOut) {
        dlBtnOut.addEventListener('click', (e) => {
            e.stopPropagation();
            const img = outBody.querySelector('img.modern-out-img');
            if (img && img.src) downloadImage(img.src, 'output_result.png');
        });
    }
    if (dlBtnPrev) {
        dlBtnPrev.addEventListener('click', (e) => {
            e.stopPropagation();
            const img = prevBody.querySelector('img.modern-out-img');
            if (img && img.src) downloadImage(img.src, 'input_preview.png');
        });
    }

    function syncImages() {
        const resultImg = resultContainer.querySelector('img');
        if (resultImg && resultImg.src) {
            if (outPh) outPh.style.display = 'none';
            let existing = outBody.querySelector('img.modern-out-img');
            if (!existing) {
                existing = document.createElement('img');
                existing.className = 'modern-out-img';
                outBody.appendChild(existing);
            }
            if (existing.src !== resultImg.src) {
                existing.src = resultImg.src;
                if (dlBtnOut) dlBtnOut.classList.add('visible');
                if (window.__hideLoaders) window.__hideLoaders();
            }
        }
        const previewImg = previewContainer.querySelector('img');
        if (previewImg && previewImg.src) {
            if (prevPh) prevPh.style.display = 'none';
            let existing2 = prevBody.querySelector('img.modern-out-img');
            if (!existing2) {
                existing2 = document.createElement('img');
                existing2.className = 'modern-out-img';
                prevBody.appendChild(existing2);
            }
            if (existing2.src !== previewImg.src) {
                existing2.src = previewImg.src;
                if (dlBtnPrev) dlBtnPrev.classList.add('visible');
            }
        }
    }

    const observer = new MutationObserver(syncImages);
    observer.observe(resultContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    observer.observe(previewContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    setInterval(syncImages, 800);
}
watchOutputs();

function watchDimensions() {
    const wContainer = document.getElementById('gradio-width');
    const hContainer = document.getElementById('gradio-height');
    const wSlider = document.getElementById('custom-width');
    const hSlider = document.getElementById('custom-height');
    const wVal = document.getElementById('custom-width-val');
    const hVal = document.getElementById('custom-height-val');
    if (!wContainer || !hContainer || !wSlider || !hSlider) {
        setTimeout(watchDimensions, 500);
        return;
    }
    function syncDims() {
        const wInput = wContainer.querySelector('input[type="range"],input[type="number"]');
        const hInput = hContainer.querySelector('input[type="range"],input[type="number"]');
        if (wInput && wInput.value) { wSlider.value = wInput.value; if(wVal) wVal.textContent = wInput.value; }
        if (hInput && hInput.value) { hSlider.value = hInput.value; if(hVal) hVal.textContent = hInput.value; }
    }
    const obs = new MutationObserver(syncDims);
    obs.observe(wContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['value']});
    obs.observe(hContainer, {childList:true, subtree:true, attributes:true, attributeFilter:['value']});
    setInterval(syncDims, 1000);
}
watchDimensions();
}
"""

DOWNLOAD_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-5-5h3V4h4v7h3l-5 5z"/><path d="M20 18H4v2h16v-2z"/></svg>'

with gr.Blocks() as demo:

    hidden_image_b64 = gr.Textbox(
        elem_id="hidden-image-b64",
        elem_classes="hidden-input",
        container=False
    )
    boxes_json = gr.Textbox(
        value="[]",
        elem_id="boxes-json-input",
        elem_classes="hidden-input",
        container=False
    )
    prompt = gr.Textbox(
        value=DEFAULT_PROMPT,
        elem_id="prompt-gradio-input",
        elem_classes="hidden-input",
        container=False
    )
    seed = gr.Slider(
        minimum=0, maximum=MAX_SEED, step=1, value=0,
        elem_id="gradio-seed",
        elem_classes="hidden-input",
        container=False
    )
    randomize_seed = gr.Checkbox(
        value=True,
        elem_id="gradio-randomize",
        elem_classes="hidden-input",
        container=False
    )
    guidance_scale = gr.Slider(
        minimum=1.0, maximum=10.0, step=0.1, value=1.0,
        elem_id="gradio-guidance",
        elem_classes="hidden-input",
        container=False
    )
    num_inference_steps = gr.Slider(
        minimum=1, maximum=20, step=1, value=4,
        elem_id="gradio-steps",
        elem_classes="hidden-input",
        container=False
    )
    height_slider = gr.Slider(
        minimum=256, maximum=2048, step=8, value=1024,
        elem_id="gradio-height",
        elem_classes="hidden-input",
        container=False
    )
    width_slider = gr.Slider(
        minimum=256, maximum=2048, step=8, value=1024,
        elem_id="gradio-width",
        elem_classes="hidden-input",
        container=False
    )

    result = gr.Image(
        elem_id="gradio-result",
        elem_classes="hidden-input",
        container=False,
        format="png"
    )
    preview = gr.Image(
        elem_id="gradio-preview",
        elem_classes="hidden-input",
        container=False
    )

    gr.HTML(f"""
    <div class="app-shell">

        <!-- Header -->
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">⌦</div>
                <span class="app-title">QIE Object Remover</span>
                <span class="app-badge">Bbox</span>
            </div>
        </div>

        <!-- Toolbar -->
        <div class="app-toolbar">
            <button id="tb-draw" class="modern-tb-btn active" title="Draw bounding boxes">
                <span class="tb-icon">▬</span><span class="tb-label">Draw</span>
            </button>
            <button id="tb-select" class="modern-tb-btn" title="Select, move, resize boxes">
                <span class="tb-icon">⇉</span><span class="tb-label">Select</span>
            </button>
            <button id="tb-reset" class="modern-tb-btn" title="Reset canvas and remove image">
                <span class="tb-icon">⟲</span><span class="tb-label">Reset</span>
            </button>
            <div class="tb-sep"></div>
            <button id="tb-del" class="modern-tb-btn" title="Delete selected box">
                <span class="tb-icon">✕</span><span class="tb-label">Delete</span>
            </button>
            <button id="tb-undo" class="modern-tb-btn" title="Undo last box">
                <span class="tb-icon">↩</span><span class="tb-label">Undo</span>
            </button>
            <button id="tb-clear" class="modern-tb-btn" title="Clear all boxes">
                <span class="tb-icon">✖</span><span class="tb-label">Clear</span>
            </button>
            <div class="tb-sep"></div>
            <button id="tb-change-img" class="modern-tb-btn" title="Upload a different image">
                <span class="tb-label">Upload…</span>
            </button>
        </div>

        <!-- Main Content -->
        <div class="app-main-row">

            <!-- Left: Canvas -->
            <div class="app-main-left">
                <div id="bbox-draw-wrap">
                    <div id="upload-prompt" class="upload-prompt-modern">
                        <div id="upload-click-area" class="upload-click-area">
                            <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="8" y="14" width="64" height="52" rx="6" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 3"/>
                                <polygon points="12,62 30,40 42,50 54,34 68,62" fill="rgba(99,102,241,0.15)" stroke="#6366f1" stroke-width="1.5"/>
                                <circle cx="28" cy="30" r="6" fill="rgba(99,102,241,0.2)" stroke="#6366f1" stroke-width="1.5"/>
                            </svg>
                        </div>
                    </div>
                    <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />
                    <canvas id="bbox-draw-canvas" width="512" height="400"></canvas>
                    <div id="bbox-status"></div>
                    <div id="bbox-count"></div>
                </div>

                <div class="hint-bar">
                    <b>Draw:</b> Click &amp; drag to create selection boxes &nbsp;·&nbsp;
                    <b>Select:</b> Click a box to move or resize &nbsp;·&nbsp;
                    <kbd>Delete</kbd> removes selected &nbsp;·&nbsp;
                    <kbd>Clear</kbd> removes all &nbsp;·&nbsp;
                    <kbd>Reset</kbd> removes image
                </div>

                <div class="json-panel">
                    <div class="json-panel-title">Bounding Boxes</div>
                    <div class="json-panel-content" id="bbox-json-content">[
  // No bounding boxes defined
]</div>
                </div>
            </div>

            <!-- Right: Controls & Output -->
            <div class="app-main-right">

                <div class="panel-card">
                    <div class="panel-card-title">Edit Instruction</div>
                    <div class="panel-card-body">
                        <label class="modern-label" for="custom-prompt-input">Prompt</label>
                        <textarea id="custom-prompt-input" class="modern-textarea" rows="2" placeholder="Describe the edit...">Remove the red highlighted object from the scene</textarea>
                    </div>
                </div>

                <div style="padding:12px 20px;">
                    <button id="custom-run-btn" class="btn-run">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 7l-7 5-7-5V5l7 5 7-5v2zm0 6l-7 5-7-5v-2l7 5 7-5v2z"/></svg>
                        Remove Object
                    </button>
                </div>

                <div class="output-frame" style="flex:1">
                    <div class="out-title">
                        <span>Output</span>
                        <span id="dl-btn-output" class="out-download-btn" title="Download">
                            {DOWNLOAD_SVG} Save
                        </span>
                    </div>
                    <div class="out-body" id="output-image-container">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Processing image…</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="output-placeholder">Result will appear here</div>
                    </div>
                </div>

                <div class="output-frame">
                    <div class="out-title">
                        <span>Input Preview</span>
                        <span id="dl-btn-preview" class="out-download-btn" title="Download">
                            {DOWNLOAD_SVG} Save
                        </span>
                    </div>
                    <div class="out-body" id="preview-image-container">
                        <div class="modern-loader" id="preview-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Preparing input…</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="preview-placeholder">Preview will appear here</div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>Seed</label>
                            <input type="range" id="custom-seed" min="0" max="2147483647" step="1" value="0">
                            <span class="slider-val" id="custom-seed-val">0</span>
                        </div>
                        <div class="checkbox-row">
                            <input type="checkbox" id="custom-randomize" checked>
                            <label for="custom-randomize">Randomize seed</label>
                        </div>
                        <div class="slider-row">
                            <label>Guidance</label>
                            <input type="range" id="custom-guidance" min="1" max="10" step="0.1" value="1.0">
                            <span class="slider-val" id="custom-guidance-val">1.0</span>
                        </div>
                        <div class="slider-row">
                            <label>Steps</label>
                            <input type="range" id="custom-steps" min="1" max="20" step="1" value="4">
                            <span class="slider-val" id="custom-steps-val">4</span>
                        </div>
                        <div class="slider-row">
                            <span class="dim-label">Width</span>
                            <input type="range" id="custom-width" min="256" max="2048" step="8" value="1024">
                            <span class="slider-val" id="custom-width-val">1024</span>
                        </div>
                        <div class="slider-row">
                            <span class="dim-label">Height</span>
                            <input type="range" id="custom-height" min="256" max="2048" step="8" value="1024">
                            <span class="slider-val" id="custom-height-val">1024</span>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <!-- Status Bar -->
        <div class="app-statusbar">
            <div class="sb-section" id="bbox-debug-count">No boxes drawn</div>
            <div class="sb-section sb-fixed">Ready</div>
        </div>

    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=None, js=bbox_drawer_js)
    demo.load(fn=None, js=wire_outputs_js)

    run_btn.click(
        fn=infer_object_removal,
        inputs=[hidden_image_b64, boxes_json, prompt, seed, randomize_seed,
                guidance_scale, num_inference_steps, height_slider, width_slider],
        outputs=[result, seed, preview],
        js="""(b64, bj, p, s, rs, gs, nis, h, w) => {
            const boxes = window.__bboxBoxes || [];
            const json = JSON.stringify(boxes);
            return [b64, json, p, s, rs, gs, nis, h, w];
        }""",
    )

    hidden_image_b64.change(
        fn=update_dimensions_on_upload,
        inputs=[hidden_image_b64],
        outputs=[width_slider, height_slider],
    )

if __name__ == "__main__":
    demo.launch(
        css=css,
        mcp_server=True, ssr_mode=False, show_error=True
    )
