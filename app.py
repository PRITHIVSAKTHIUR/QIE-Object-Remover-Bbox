import gradio as gr
import numpy as np
import random
import torch
import spaces
import base64
from io import BytesIO
from typing import Iterable
from PIL import Image, ImageDraw
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.purple = colors.Color(
    name="purple",
    c50="#FAF5FF",
    c100="#F3E8FF",
    c200="#E9D5FF",
    c300="#DAB2FF",
    c400="#C084FC",
    c500="#A855F7",
    c600="#9333EA",
    c700="#7E22CE",
    c800="#6B21A8",
    c900="#581C87",
    c950="#3B0764",
)


class PurpleTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.purple,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


purple_theme = PurpleTheme()

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
        "repo": "prithivMLmods/QIE-2509-Object-Remover-Bbox",
        "weights": "QIE-2509-Object-Remover-Bbox-5000.safetensors",
        "adapter_name": "object-remover",
    },
}
loaded = False

DEFAULT_PROMPT = "Remove the red highlighted object from the scene"


def b64_to_pil(b64_str: str) -> Image.Image | None:
    """Helper to decode base64 string from JS into a PIL Image"""
    if not b64_str or not b64_str.startswith("data:image"):
        return None
    try:
        _, data = b64_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def burn_boxes_onto_image(pil_image: Image.Image, boxes_json_str: str) -> Image.Image:
    """Burn red outline-only rectangles onto the image (no fill)."""
    import json
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
        # Red outline only — no fill
        draw.rectangle([lx, ty, rx, by_], outline=(255, 0, 0), width=bw)

    return img


@spaces.GPU
def infer_object_removal(
    b64_str: str,
    boxes_json: str,
    prompt: str,
    seed: int = 0,
    randomize_seed: bool = True,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 4,
    height: int = 1024,
    width: int = 1024,
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
    print(f"Prompt: {prompt}")
    print(f"Boxes JSON received: '{boxes_json}'")

    source_image = b64_to_pil(b64_str)
    if source_image is None:
        raise gr.Error("Please upload an image first using the Bbox editor area.")

    import json
    try:
        boxes = json.loads(boxes_json) if boxes_json and boxes_json.strip() else []
    except Exception as e:
        print(f"JSON parse error: {e}")
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


def update_dimensions_on_upload(b64_str: str):
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
body,.gradio-container{background-color:#FAF5FF!important;background-image:linear-gradient(#E9D5FF 1px,transparent 1px),linear-gradient(90deg,#E9D5FF 1px,transparent 1px)!important;background-size:40px 40px!important;font-family:'Outfit',sans-serif!important}
.dark body,.dark .gradio-container{background-color:#1a1a1a!important;background-image:linear-gradient(rgba(168,85,247,.1) 1px,transparent 1px),linear-gradient(90deg,rgba(168,85,247,.1) 1px,transparent 1px)!important;background-size:40px 40px!important}
#col-container{margin:0 auto;max-width:1200px}
#main-title{text-align:center!important;padding:1rem 0 .5rem 0}
#main-title h1{font-size:2.4em!important;font-weight:700!important;background:linear-gradient(135deg,#A855F7 0%,#C084FC 50%,#9333EA 100%);background-size:200% 200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:gradient-shift 4s ease infinite;letter-spacing:-.02em}
@keyframes gradient-shift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
#subtitle{text-align:center!important;margin-bottom:1.5rem}
#subtitle p{margin:0 auto;color:#666;font-size:1rem;text-align:center!important}
#subtitle a{color:#A855F7!important;text-decoration:none;font-weight:500}
#subtitle a:hover{text-decoration:underline}
.gradio-group{background:rgba(255,255,255,.9)!important;border:2px solid #E9D5FF!important;border-radius:12px!important;box-shadow:0 4px 24px rgba(168,85,247,.08)!important;backdrop-filter:blur(10px);transition:all .3s ease}
.gradio-group:hover{box-shadow:0 8px 32px rgba(168,85,247,.12)!important;border-color:#C084FC!important}
.dark .gradio-group{background:rgba(30,30,30,.9)!important;border-color:rgba(168,85,247,.3)!important}
.primary{border-radius:8px!important;font-weight:600!important;letter-spacing:.02em!important;transition:all .3s ease!important}
.primary:hover{transform:translateY(-2px)!important}
.gradio-textbox textarea{font-family:'IBM Plex Mono',monospace!important;font-size:.95rem!important;line-height:1.7!important;background:rgba(255,255,255,.95)!important;border:1px solid #E9D5FF!important;border-radius:8px!important}
.gradio-accordion{border-radius:10px!important;border:1px solid #E9D5FF!important}
.gradio-accordion>.label-wrap{background:rgba(168,85,247,.03)!important;border-radius:10px!important}
footer{display:none!important}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.gradio-row{animation:fadeIn .4s ease-out}
label{font-weight:600!important;color:#333!important}
.dark label{color:#eee!important}
.gradio-slider input[type="range"]{accent-color:#A855F7!important}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:rgba(168,85,247,.05);border-radius:4px}
::-webkit-scrollbar-thumb{background:linear-gradient(135deg,#A855F7,#C084FC);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(135deg,#9333EA,#A855F7)}

#bbox-draw-wrap{position:relative;border:2px dashed #C084FC;border-radius:12px;overflow:hidden;background:#1a1a1a;min-height:420px;transition: border-color 0.2s ease;}
#bbox-draw-wrap:hover{border-color:#A855F7}
#bbox-draw-canvas{cursor:crosshair;display:block;margin:0 auto}
.bbox-hint{background:rgba(168,85,247,.08);border:1px solid #E9D5FF;border-radius:8px;padding:10px 16px;margin:8px 0;font-size:.9rem;color:#6B21A8}
.dark .bbox-hint{background:rgba(168,85,247,.15);border-color:rgba(168,85,247,.3);color:#C084FC}

/* Custom Uiverse.io Upload Prompt Component styling */
.upload-container {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    z-index: 20;
    height: 300px;
    width: 300px;
    border-radius: 10px;
    box-shadow: 4px 4px 30px rgba(168, 85, 247, 0.2);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 10px;
    gap: 5px;
    background-color: rgba(250, 245, 255, 0.95);
    backdrop-filter: blur(8px);
}
.dark .upload-container {
    background-color: rgba(30, 30, 30, 0.95);
    box-shadow: 4px 4px 30px rgba(0, 0, 0, 0.5);
}

.upload-header {
    flex: 1;
    width: 100%;
    border: 2px dashed #A855F7;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    cursor: pointer;
    transition: background-color 0.2s;
    color: #A855F7;
}
.upload-header:hover {
    background-color: rgba(168, 85, 247, 0.05);
}
.dark .upload-header:hover {
    background-color: rgba(168, 85, 247, 0.15);
}

.upload-header svg {
    height: 100px;
}

.upload-header p {
    text-align: center;
    color: #6B21A8;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    margin-top: 10px;
}
.dark .upload-header p {
    color: #DAB2FF;
}

.upload-footer {
    background-color: rgba(168, 85, 247, 0.08);
    width: 100%;
    height: 40px;
    padding: 8px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    color: #6B21A8;
    border: none;
    box-sizing: border-box;
}
.dark .upload-footer {
    background-color: rgba(168, 85, 247, 0.15);
    color: #DAB2FF;
}

.upload-footer svg {
    height: 130%;
    fill: #A855F7;
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    padding: 2px;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(168, 85, 247, 0.2);
    transition: transform 0.2s;
}
.dark .upload-footer svg {
    background-color: rgba(0, 0, 0, 0.3);
}
.upload-footer svg:hover {
    transform: scale(1.1);
}

.upload-footer p {
    flex: 1;
    text-align: center;
    font-family: 'Outfit', sans-serif;
    font-size: 0.9rem;
    margin: 0;
}

.bbox-toolbar-section{
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    justify-content:center;
    align-items:center;
    padding:12px 16px;
    margin-top:10px;
    background:rgba(255,255,255,.92);
    border:2px solid #E9D5FF;
    border-radius:10px;
    box-shadow:0 2px 12px rgba(168,85,247,.08);
}
.dark .bbox-toolbar-section{
    background:rgba(30,30,30,.9);
    border-color:rgba(168,85,247,.3);
}
.bbox-toolbar-section .toolbar-label{
    font-family:'Outfit',sans-serif;
    font-weight:600;
    font-size:13px;
    color:#6B21A8;
    margin-right:6px;
    user-select:none;
}
.dark .bbox-toolbar-section .toolbar-label{color:#C084FC}
.bbox-toolbar-section .toolbar-divider{
    width:1px;
    height:28px;
    background:#E9D5FF;
    margin:0 4px;
}
.dark .bbox-toolbar-section .toolbar-divider{background:rgba(168,85,247,.3)}
.bbox-toolbar-section button{
    color:#fff;
    border:none;
    padding:7px 15px;
    border-radius:7px;
    cursor:pointer;
    font-family:'Outfit',sans-serif;
    font-weight:600;
    font-size:13px;
    box-shadow:0 2px 5px rgba(0,0,0,.15);
    transition:background .2s,transform .15s,box-shadow .2s;
}
.bbox-toolbar-section button:hover{transform:translateY(-1px);box-shadow:0 4px 10px rgba(0,0,0,.2)}
.bbox-toolbar-section button:active{transform:translateY(0)}
.bbox-tb-draw{background:#9333EA}
.bbox-tb-draw:hover{background:#A855F7}
.bbox-tb-draw.active{background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,.5)}
.bbox-tb-select{background:#6366f1}
.bbox-tb-select:hover{background:#818cf8}
.bbox-tb-select.active{background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,.5)}
.bbox-tb-del{background:#dc2626}
.bbox-tb-del:hover{background:#ef4444}
.bbox-tb-undo{background:#7E22CE}
.bbox-tb-undo:hover{background:#9333EA}
.bbox-tb-clear{background:#be123c}
.bbox-tb-clear:hover{background:#e11d48}
.bbox-tb-change{background:#4b5563}
.bbox-tb-change:hover{background:#6b7280}

#bbox-status{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.75);color:#00ff88;padding:5px 10px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:11px;z-index:10;display:none;pointer-events:none}
#bbox-count{position:absolute;top:10px;right:10px;background:rgba(147,51,234,.85);color:#fff;padding:4px 10px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:11px;z-index:10;display:none}

#bbox-debug-count{
    text-align:center;
    padding:6px 12px;
    margin-top:6px;
    font-family:'IBM Plex Mono',monospace;
    font-size:12px;
    color:#6B21A8;
    background:rgba(168,85,247,.06);
    border:1px dashed #C084FC;
    border-radius:6px;
}
.dark #bbox-debug-count{color:#C084FC;background:rgba(168,85,247,.12)}

.hidden-input {
    display: none !important;
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

        const btnDraw    = document.getElementById('tb-draw');
        const btnSelect  = document.getElementById('tb-select');
        const btnDel     = document.getElementById('tb-del');
        const btnUndo    = document.getElementById('tb-undo');
        const btnClear   = document.getElementById('tb-clear');
        const btnChange  = document.getElementById('tb-change-img');
        
        const uploadPrompt = document.getElementById('upload-prompt');
        const uploadHeader = document.getElementById('upload-header');
        const fileInput    = document.getElementById('custom-file-input');

        if (!canvas || !wrap || !debugCount || !btnDraw || !fileInput) {
            console.log('[BBox] waiting for DOM...');
            setTimeout(initCanvasBbox, 250);
            return;
        }

        window.__bboxInitDone = true;
        console.log('[BBox] canvas init OK');
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
        const HANDLE  = 7;
        const RED_STROKE = 'rgba(255,0,0,0.95)';
        const RED_STROKE_WIDTH = 3;
        const SEL_STROKE = 'rgba(0,120,255,0.95)';

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

        function syncToGradio() {
            window.__bboxBoxes = boxes;
            const jsonStr = JSON.stringify(boxes);

            if (debugCount) {
                debugCount.textContent = boxes.length > 0
                    ? '\u2705 ' + boxes.length + ' box' + (boxes.length > 1 ? 'es' : '') +
                      ' ready  |  JSON: ' + jsonStr.substring(0,80) +
                      (jsonStr.length > 80 ? '\u2026' : '')
                    : '\u2B1C No boxes drawn yet';
            }

            const container = document.getElementById('boxes-json-input');
            if (!container) return;
            const targets = [
                ...container.querySelectorAll('textarea'),
                ...container.querySelectorAll('input:not([type="file"])')
            ];
            targets.forEach(el => {
                const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                const ns = Object.getOwnPropertyDescriptor(proto, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, jsonStr);
                    el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        }

        function syncImageToGradio(dataUrl) {
            const container = document.getElementById('hidden-image-b64');
            if (!container) return;
            const targets = [
                ...container.querySelectorAll('textarea'),
                ...container.querySelectorAll('input')
            ];
            targets.forEach(el => {
                const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                const ns = Object.getOwnPropertyDescriptor(proto, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, dataUrl);
                    el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        }

        function redraw(tempRect) {
            ctx.clearRect(0,0,dispW,dispH);
            if (!baseImg) { 
                ctx.fillStyle='#1a1a1a'; ctx.fillRect(0,0,dispW,dispH);
                updateBadge(); return; 
            }
            ctx.drawImage(baseImg, 0, 0, dispW, dispH);

            boxes.forEach((b,i) => {
                const p = n2px(b);
                const lx=p.x1, ty=p.y1, w=p.x2-p.x1, h=p.y2-p.y1;

                /* RED OUTLINE ONLY — no fill */
                if (i === selectedIdx) {
                    ctx.strokeStyle = SEL_STROKE;
                    ctx.lineWidth = RED_STROKE_WIDTH + 1;
                    ctx.setLineDash([6,3]);
                } else {
                    ctx.strokeStyle = RED_STROKE;
                    ctx.lineWidth = RED_STROKE_WIDTH;
                    ctx.setLineDash([]);
                }
                ctx.strokeRect(lx, ty, w, h);
                ctx.setLineDash([]);

                /* label tag */
                ctx.fillStyle = i===selectedIdx ? 'rgba(0,120,255,0.85)' : 'rgba(255,0,0,0.85)';
                ctx.font = 'bold 11px IBM Plex Mono,monospace';
                ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                const label = '#'+(i+1);
                const tw = ctx.measureText(label).width;
                ctx.fillRect(lx, ty-16, tw+6, 16);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, lx+3, ty-14);

                if (i === selectedIdx) drawHandles(p);
            });

            /* temp drawing rect — outline only */
            if (tempRect) {
                const rx = Math.min(tempRect.x1,tempRect.x2);
                const ry = Math.min(tempRect.y1,tempRect.y2);
                const rw = Math.abs(tempRect.x2-tempRect.x1);
                const rh = Math.abs(tempRect.y2-tempRect.y1);
                ctx.strokeStyle = RED_STROKE;
                ctx.lineWidth = RED_STROKE_WIDTH;
                ctx.setLineDash([6,3]);
                ctx.strokeRect(rx, ry, rw, rh);
                ctx.setLineDash([]);
            }
            updateBadge();
        }

        function drawHandles(p) {
            const pts = handlePoints(p);
            ctx.fillStyle = 'rgba(0,120,255,0.9)';
            ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
            for (const k in pts) {
                const h = pts[k];
                ctx.fillRect(h.x-HANDLE, h.y-HANDLE, HANDLE*2, HANDLE*2);
                ctx.strokeRect(h.x-HANDLE, h.y-HANDLE, HANDLE*2, HANDLE*2);
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
                        showStatus('Resizing box #'+(selectedIdx+1));
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
                        showStatus('Resizing box #'+(selectedIdx+1));
                        redraw(); return;
                    }
                    dragging = true; dragType = 'move';
                    dragStart = {x, y};
                    dragOrig = {...boxes[selectedIdx]};
                    showStatus('Moving box #'+(selectedIdx+1));
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
                    console.log('[BBox] created box #'+boxes.length, nb);
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

        // --- File Upload Logic ---
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

        uploadHeader.addEventListener('click', () => fileInput.click());
        btnChange.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            processFile(e.target.files[0]);
            e.target.value = ''; // Reset input to allow re-upload of same file
        });

        wrap.addEventListener('dragover', (e) => {
            e.preventDefault();
            wrap.style.borderColor = '#A855F7';
            wrap.style.boxShadow = '0 0 15px rgba(168,85,247,0.3)';
        });
        wrap.addEventListener('dragleave', (e) => {
            e.preventDefault();
            wrap.style.borderColor = '';
            wrap.style.boxShadow = '';
        });
        wrap.addEventListener('drop', (e) => {
            e.preventDefault();
            wrap.style.borderColor = '';
            wrap.style.boxShadow = '';
            if (e.dataTransfer.files.length) {
                processFile(e.dataTransfer.files[0]);
            }
        });

        // --- Toolbar Logic ---
        btnDraw.addEventListener('click',   ()=>setMode('draw'));
        btnSelect.addEventListener('click', ()=>setMode('select'));

        btnDel.addEventListener('click', () => {
            if (selectedIdx >= 0 && selectedIdx < boxes.length) {
                const removed = selectedIdx + 1;
                boxes.splice(selectedIdx, 1);
                window.__bboxBoxes = boxes;
                selectedIdx = -1;
                syncToGradio(); redraw();
                showStatus('Box #'+removed+' deleted');
            } else {
                showStatus('No box selected');
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


with gr.Blocks() as demo:
    gr.Markdown("# **QIE-Object-Remover-Bbox**", elem_id="main-title")
    gr.Markdown(
        "Perform diverse image edits using a specialized [LoRA](https://huggingface.co/prithivMLmods/QIE-2509-Object-Remover-Bbox). "
        "Upload an image directly into the bounding box editor area below, draw red bounding boxes over the objects you want to remove, and click Remove Object. "
        "Multiple boxes supported. Select, move, resize or delete individual boxes. Open on [GitHub](https://github.com/PRITHIVSAKTHIUR/QIE-Object-Remover-Bbox)",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            
            hidden_image_b64 = gr.Textbox(
                elem_id="hidden-image-b64", 
                elem_classes="hidden-input", 
                container=False
            )

            #gr.Markdown("### **Bbox Edit Controller**")

            gr.HTML(
                """
                <div id="bbox-draw-wrap">
                    <div id="upload-prompt" class="upload-container">
                      <div class="upload-header" id="upload-header">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M7 10V9C7 6.23858 9.23858 4 12 4C14.7614 4 17 6.23858 17 9V10C19.2091 10 21 11.7909 21 14C21 15.4806 20.1956 16.8084 19 17.5M7 10C4.79086 10 3 11.7909 3 14C3 15.4806 3.8044 16.8084 5 17.5M7 10C7.43285 10 7.84965 10.0688 8.24006 10.1959M12 12V21M12 12L15 15M12 12L9 15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                        <p>Browse File to upload!</p>
                      </div>
                      <div class="upload-footer">
                        <svg fill="currentColor" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                            <path d="M15.331 6H8.5v20h15V14.154h-8.169z"></path>
                            <path d="M18.153 6h-.009v5.342H23.5v-.002z"></path>
                        </svg>
                        <p>Not selected file</p>
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M5.16565 10.1534C5.07629 8.99181 5.99473 8 7.15975 8H16.8402C18.0053 8 18.9237 8.9918 18.8344 10.1534L18.142 19.1534C18.0619 20.1954 17.193 21 16.1479 21H7.85206C6.80699 21 5.93811 20.1954 5.85795 19.1534L5.16565 10.1534Z" stroke="currentColor" stroke-width="2"></path>
                            <path d="M19.5 5H4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
                            <path d="M10 3C10 2.44772 10.4477 2 11 2H13C13.5523 2 14 2.44772 14 3V5H10V3Z" stroke="currentColor" stroke-width="2"></path>
                        </svg>
                      </div>
                      <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />
                    </div>
                    
                    <canvas id="bbox-draw-canvas" width="512" height="400"></canvas>
                    <div id="bbox-status"></div>
                    <div id="bbox-count"></div>
                </div>
                """
            )

            gr.HTML(
                """
                <div class="bbox-toolbar-section">
                    <span class="toolbar-label">🛠 Tools:</span>
                    <button id="tb-draw"   class="bbox-tb-draw active"  title="Draw new boxes">✏️ Draw</button>
                    <button id="tb-select" class="bbox-tb-select"       title="Select / move / resize">🔲 Select</button>
                    <div class="toolbar-divider"></div>
                    <span class="toolbar-label">Actions:</span>
                    <button id="tb-del"    class="bbox-tb-del"          title="Delete selected box">✕ Delete</button>
                    <button id="tb-undo"   class="bbox-tb-undo"         title="Remove last box">↩ Undo</button>
                    <button id="tb-clear"  class="bbox-tb-clear"        title="Remove all boxes">🗑 Clear All</button>
                    <div class="toolbar-divider"></div>
                    <button id="tb-change-img" class="bbox-tb-change"   title="Upload a different image">📸 Change Image</button>
                </div>
                """
            )

            gr.HTML('<div id="bbox-debug-count">\u2B1C No boxes drawn yet</div>')

            boxes_json = gr.Textbox(
                value="[]",
                elem_id="boxes-json-input",
                elem_classes="hidden-input",
                container=False
            )
            
            gr.HTML(
                '<div class="bbox-hint">'
                "<b>Draw mode:</b> Click & drag to create red rectangles. "
                "<b>Select mode:</b> Click a box to select it \u2192 drag to <b>move</b>, "
                "drag handles to <b>resize</b>. Use <b>Delete Selected</b> to remove one box."
                "</div>"
            )
            
            prompt = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=1,
                info="Edit the prompt if needed",
            )

            run_btn = gr.Button("\U0001F5D1\uFE0F Remove Object", variant="primary", size="lg")

        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", height=475, format="png")
            preview = gr.Image(label="Input Sent to Model (with red boxes)", height=415)

            with gr.Accordion("Advanced Settings", open=False, visible=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=20, step=1, value=4)
                with gr.Row():
                    height_slider = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024)
                    width_slider = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024)

    demo.load(fn=None, js=bbox_drawer_js)

    run_btn.click(
        fn=infer_object_removal,
        inputs=[hidden_image_b64, boxes_json, prompt, seed, randomize_seed,
                guidance_scale, num_inference_steps, height_slider, width_slider],
        outputs=[result, seed, preview],
        js="""(b64, bj, p, s, rs, gs, nis, h, w) => {
            const boxes = window.__bboxBoxes || [];
            const json  = JSON.stringify(boxes);
            console.log('[BBox] submitting', boxes.length, 'boxes:', json);
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
        css=css, theme=purple_theme,
        mcp_server=True, ssr_mode=False, show_error=True
    )
