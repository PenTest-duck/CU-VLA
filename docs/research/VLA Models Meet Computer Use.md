# VLA models meet computer use: the convergence of robotics AI and GUI automation

**Vision-Language-Action models and computer-use agents are converging on the same architectural pattern — perceive screen, reason, act — but from opposite directions.** Robotics VLAs like pi0 and RT-2 master continuous high-frequency control of physical actuators, while GUI agents like UI-TARS and ShowUI master discrete click-and-type interactions on digital screens. No one has yet bridged these fully: treating general computer interaction as continuous control at 30–60 Hz, the way robots control joints. But the pieces exist — small models that fit on a MacBook, world models that predict UI state transitions, and Tesla's newly announced Digital Optimus project betting billions that FSD neural networks can drive computers the way they drive cars. This report maps the full landscape across six dimensions.

---

## 1. GUI agents are independently reinventing VLA architectures

The most striking finding is one of **convergent evolution**: GUI automation researchers have arrived at VLA-like architectures without descending from robotics VLAs. Models like ShowUI, UI-TARS, and SeeClick follow the same vision → language reasoning → action output pattern as RT-2 or pi0, but were developed independently for screen-based interaction.

**ShowUI** (CVPR 2025, Outstanding Paper at NeurIPS 2024 workshop) is the most explicit "GUI-VLA." Built on Qwen2-VL-2B, it explicitly identifies as a "Vision-Language-Action model for GUI agents." Its key innovation is UI-Guided Visual Token Selection — building a component graph from ~1,296 image patches to identify ~167 interactive elements, achieving sub-50ms element localization. **UI-TARS** (ByteDance) is architecturally the closest to a true end-to-end VLA among the major agents, available at 2B, 7B, and 72B parameters, processing screenshots only and outputting actions directly via trained System-2 reasoning. It outperforms GPT-4o and Claude 3.5 on most GUI benchmarks. **SeeClick** (ACL 2024) pioneered the VLA-for-GUIs approach with GUI grounding pre-training on Common Crawl data.

Meanwhile, the "big lab" computer-use agents take a fundamentally different approach — bolting tool-use capabilities onto general-purpose VLMs:

| Agent | Architecture | Input | Action frequency | Key differentiator |
|-------|-------------|-------|-----------------|-------------------|
| **Claude Computer Use** | General VLM + tool calls | Screenshots | ~0.2–0.5 Hz | Full desktop access, schema-less tools |
| **OpenAI CUA/Operator** | GPT-4o + RL training | Screenshots | ~0.2–0.5 Hz | Chain-of-thought reasoning, 38.1% on OSWorld |
| **Project Mariner** | Gemini 2.5 Pro | Pixels + web elements | Variable | "Teach & Repeat" demonstration learning, 83.5% on WebVoyager |
| **UI-TARS** | Purpose-built VLM | Screenshots only | ~0.2–0.5 Hz | End-to-end, no DOM/a11y dependency |
| **ShowUI** | GUI-VLA (2B) | Screenshots only | Sub-50ms grounding | Smallest, fastest dedicated model |
| **Microsoft Fara-7B** | Native CUA model | Screenshots | ~0.2–0.5 Hz | 1/10th output tokens vs. SoM agents |

**No robotics VLA has been adapted for GUI use.** Pi0, RT-2, Octo, OpenVLA, and GR00T remain exclusively in physical manipulation. The domains share architectural DNA but differ on three axes: action space (continuous 6-DoF vs. discrete click/type), control frequency (**5–50 Hz** vs. 0.2–1 Hz), and environment dimensionality (3D physical vs. 2D digital). The "pure vision" trend is notable — the most advanced GUI agents process screenshots only, avoiding DOM or accessibility APIs, mirroring how robotics VLAs use raw camera input.

---

## 2. High-frequency continuous control remains the unexplored frontier

The core idea — decomposing computer actions to primitives like `mouse_down`, `mouse_up`, `key_down('h')`, `key_up('h')` and running at 30–60 Hz — has **no complete implementation for general computer use**, but several pieces exist at the boundary.

The closest precedent is **DeepMind's SIMA / SIMA 2** (2024–2025), which uses a "pixels in, keyboard-and-mouse out" interface across 9 commercial video games. SIMA 2, powered by Gemini, doubled success rates to ~62% on complex tasks. The philosophy is exactly right — generic human-like interface, no game APIs — but the control frequency remains relatively low. **Game-TARS** (ByteDance, October 2025) takes this further with a unified action space grounded in native keyboard-mouse inputs, pre-trained on >500B tokens. Its key empirical finding: **unified keyboard/mouse action spaces initially underperform domain-specific ones at small data scales but show superior scalability** — performance kept improving with data while specialized action spaces plateaued.

**JARVIS-VLA** (ACL Findings 2025) is the most explicit robotics-to-computer bridge. It applies the VLA paradigm directly to Minecraft control, discretizing mouse X/Y into 21 bins using mu-law discretization (borrowed from audio/robotics), with 29 keyboard tokens and 22 mouse tokens. It achieves **~55 FPS on 4×A800 GPUs**, targeting >40 Hz for human-level gameplay. The "Running VLAs at Real-time Speed" paper (arXiv:2510.26742) demonstrates that pi0-level VLAs can run at **30 Hz frame rate and up to 480 Hz trajectory frequency** on a single consumer GPU via streaming inference — separating the compute-bound VLM backbone from the IO-bound action expert on concurrent CUDA streams.

**Action chunking** is the critical insight connecting these results. Rather than running the full model at 60 Hz, you predict N future actions in one forward pass (SmolVLA predicts 50 at once), then execute them at high frequency while the next observation processes. This decouples inference frequency from control frequency — a 3 Hz model can drive 30+ Hz control.

The hierarchical computer control paper (arXiv:2509.18230) explicitly lists "implementing continuous control for more accurate mouse control" as future work, confirming this is a recognized open direction. One paper does treat mouse control as literally continuous: an Active Inference model (arXiv:2510.14611) formulates 1D mouse pointing with continuous state, action, and observation spaces — but only for simple pointing, not general computer use.

Why does the gap persist? Four reasons: most computer tasks are inherently discrete and don't need 60 Hz; large VLMs providing reasoning can't run that fast; human computer interaction is variable-rate (long pauses then bursts), unlike constant-rate robot teleoperation; and hybrid GUI+API approaches that skip to API calls outperform pure low-level control on current benchmarks.

---

## 3. UI world models exploded in 2025–2026

A remarkable cluster of papers in the past 18 months has established UI state prediction as a distinct research field, with at least five competing architectural approaches.

**CUWM (Computer-Using World Model, arXiv:2602.17365)** is the most directly relevant — a world model specifically for desktop software (Microsoft Office). It uses a two-stage factorization: a textual model predicts UI state changes, then a visual model renders the predicted screenshot. The textual stage is optimized with GRPO reinforcement learning. At inference time, a frozen agent uses the world model to simulate candidate actions before execution, improving decision quality.

**ViMo** (arXiv:2504.13936, Queen Mary/Oxford/Huawei) is the first purely visual GUI world model, generating future mobile app screenshots via diffusion. It decomposes generation into graphics prediction (with symbolic text placeholders) and text content prediction, achieving **14% improvement in step-wise action accuracy** and task completion rising from 33% to 41%.

The code-based approach is gaining traction as the most practical. **Code2World** (arXiv:2602.09856) predicts next visual state by generating renderable HTML, trained on 80K screen-action pairs translated to code. **gWorld** (arXiv:2602.01576) follows the same philosophy, achieving 81.9% accuracy with virtually cost-free rendering (~0.3s). Both avoid the notorious difficulty of rendering readable text in diffusion models by generating structured code instead.

For general interactive environments, **Genie 2** (DeepMind, December 2024) generates interactive 3D worlds from single images using autoregressive latent diffusion. **DIAMOND** (NeurIPS 2024 Spotlight) trains RL agents entirely within diffusion-generated Atari environments, scoring 46% better than human with only 13M parameters. **GameNGen** (ICLR 2025) simulates DOOM at 20 FPS using fine-tuned Stable Diffusion with 4 denoising steps. These demonstrate the feasibility of neural environment simulation, though none target software UIs specifically.

**WebDreamer** (TMLR 2025) takes a different angle entirely: LLMs already encode knowledge about website structures and can serve as world models. It uses GPT-4o to simulate candidate action outcomes as text descriptions, scoring them to select the best action — achieving **33% relative improvement** over reactive agents on VisualWebArena.

| System | Modality | Environment | Key strength |
|--------|----------|-------------|-------------|
| CUWM | Text → Visual | Desktop (Office) | RL-optimized textual prediction |
| ViMo | Visual (diffusion) | Mobile apps | First pixel-level UI prediction |
| Code2World | Code (HTML) | Android | Pixel-perfect text rendering |
| gWorld | Code (web) | Mobile | 81.9% accuracy, fast rendering |
| WebDreamer | Text (LLM) | Web | No training needed, uses LLM knowledge |
| MobileDreamer | Text (sketch) | Mobile | Efficient tree-of-prediction search |

---

## 4. Tesla's Digital Optimus bets FSD can drive computers

**Tesla's Digital Optimus (codenamed "Macrohard") is a joint Tesla-xAI project announced March 11, 2026, designed to autonomously operate computers by watching screens and controlling keyboard/mouse in real time.** The name is a deliberate jab at Microsoft — Musk stated it aims to "emulate the function of entire companies."

The architecture uses a **dual-process design** mirroring Kahneman's System 1/System 2:

- **System 1 (Tesla):** Fast, reactive processing of the past 5 seconds of continuous screen video and keyboard/mouse actions, running on Tesla's **AI4 inference chip** (~$650/chip). This leverages the same end-to-end neural networks powering Full Self-Driving — the core bet being that perceiving roads transfers to perceiving screens.
- **System 2 (xAI's Grok):** Higher-level reasoning and planning, acting as "master conductor." Runs on xAI's server infrastructure, used "relatively frugally" for complex decisions.

The project uses **distillation** — massive teacher models (Grok) generate reasoned solutions that are compressed into smaller student models running locally on AI4 hardware. The deployment vision is extraordinary: Tesla plans a distributed compute network using **parked vehicles' idle AI4 chips** and Supercharger stations (~7 GW of power infrastructure) as compute hubs.

The project has had a turbulent trajectory. Macrohard began as an xAI-internal initiative (August 2025 trademark filing, Musk recruiting on X). After xAI's acquisition by SpaceX in February 2026, it became one of four xAI divisions led by co-founder Toby Pohlen — who departed just 16 days later. Tesla engineers were brought in to "rescue" the stalled project. As of April 2026, **Ashok Elluswamy** (Tesla VP of AI Software) leads Macrohard across both organizations. xAI's Colossus 2 supercomputer (bearing the MACROHARD name) is training 7 models simultaneously, including a **10-trillion-parameter model** — the largest publicly announced by any lab.

**No public demo or product exists yet.** Musk targets September 2026, though his timelines are historically optimistic by 1–3 years. The announcement has legal implications: a shareholder lawsuit alleges Musk breached fiduciary duty by building AI capabilities at xAI rather than Tesla, and the Digital Optimus announcement — proving xAI's technology was always intended for Tesla — complicates his legal position.

---

## 5. Sub-billion parameter models make on-device VLA feasible today

Several models can run VLA-style inference at **10+ Hz on a MacBook Pro M1 with 8GB RAM**, with memory budgets of ~5 GB after OS overhead.

**SmolVLA** (Hugging Face, June 2025, 450M parameters) is purpose-built for this. At **0.9 GB VRAM and 18ms per inference step** on GPU, it uses flow-matching (not autoregressive generation) to predict 50-action chunks in one forward pass. It matches or exceeds OpenVLA's 7B-parameter performance on LIBERO and real-world robot tasks. Its asynchronous inference stack specifically decouples VLM processing from action execution for real-time control. On M1, inference likely runs at **50–150ms per step** — comfortably above 10 Hz, with action chunking enabling much higher effective control frequency.

| Model | Parameters | Memory (4-bit) | 10+ Hz on M1? | Best for |
|-------|-----------|----------------|---------------|----------|
| **SmolVLA** | 450M | ~0.4 GB | ✅ Yes | Direct VLA use |
| **SmolVLM-256M** | 256M | ~0.2 GB | ✅ Yes | Minimal backbone + custom action head |
| **Moondream 0.5B** | 500M | ~0.5 GB | ✅ Yes | Edge VLM with detection/pointing |
| **Florence-2-base** | 230M | ~0.2 GB | ✅ Yes | Seq2seq backbone (natural for actions) |
| **SmolVLM-2.2B** | 2.2B | ~1.5 GB | ⚠️ Marginal | Stronger reasoning, 3–5 Hz |
| **TinyVLA** | 40M–1.4B | ~0.5 GB | ✅ Yes | Diffusion action head, 20× faster than OpenVLA |
| **Octo** | 27M/93M | <0.2 GB | ✅ Trivially | Extremely lightweight, limited capability |

The key architectural insight: **flow-matching and diffusion action heads bypass the autoregressive bottleneck.** Instead of generating action tokens one by one (slow), these models produce entire action vectors in a single forward pass. Combined with action chunking, a 7 Hz model inference rate becomes 50+ Hz effective control.

**MLX** (Apple's ML framework) is the recommended runtime, achieving **50% faster inference than Ollama** and supporting SmolVLM, Qwen2-VL, Moondream, and PaliGemma out of the box via mlx-vlm. llama.cpp now supports multimodal GGUF models (since May 2025), including SmolVLM-256M/500M, as a cross-platform alternative.

---

## 6. A working proof-of-concept in under 50 lines of Python

The simplest possible demonstration of VLA-style computer control targets a **reaction time test** — detect a color change on screen, click as fast as possible, measure latency at each pipeline stage.

**Screen capture:** MSS (Multiple Screen Shot) achieves **47–61 FPS on M1** using native CoreGraphics under the hood, compared to PyAutoGUI's catastrophic ~2.8 FPS (it shells out to the `screencapture` CLI tool). Capturing a small region (~100×100 pixels) maximizes frame rate. Install: `pip install mss`.

**Input control:** Quartz CoreGraphics provides **sub-millisecond** event posting with full control over `mouse_down` / `mouse_up` separation — exactly the first-principles decomposition needed:

```python
from Quartz.CoreGraphics import *
def mouse_down(x, y):
    event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)
def mouse_up(x, y):
    event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)
```

PyAutoGUI is simpler but adds ~110ms default delay (set `PAUSE=0` to eliminate). Both require macOS Accessibility and Screen Recording permissions.

**Target task:** A self-hosted Tkinter window changing from red to green at random intervals is simplest — no browser latency, full timing instrumentation. The Human Benchmark reaction test (humanbenchmark.com) is a good second option with built-in scoring and human comparison baselines (~200ms average). Multiple existing Python bots demonstrate the pipeline.

**The upgrade path from "kindergarten" to VLA:**

- **Level 0** — Pixel threshold: average green channel > 150 → click. Total latency: ~20–30ms. Proves the capture→decide→act pipeline.
- **Level 1** — Small CNN: Replace threshold with MobileNetV3 (~5ms on M1 GPU). Proves neural inference in the loop.
- **Level 2** — Vision-Language Model: Use SmolVLM-256M or Moondream 0.5B with a text prompt like "Is the screen green? Output click coordinates." Latency: ~100–200ms. Proves VLM-driven action selection.
- **Level 3** — Full VLA: Use SmolVLA (450M) or attach a flow-matching action head to SmolVLM. Predict action chunks. This is a genuine VLA controlling a computer at >10 Hz.

**Latency budget per component:** Screen capture ~15–25ms (MSS), preprocessing ~1–3ms (resize+normalize), model inference ~5–500ms (depends on model), action decision <1ms, action execution <1ms. The total ranges from ~25ms (no NN) to ~530ms (large VLM), with the sweet spot around **50–150ms** for small models at 10+ Hz.

**Key existing repos:** ShowUI-Aloha (github.com/showlab/ShowUI-Aloha) records human demonstrations for computer-use agents on macOS; Pine (github.com/petercunha/Pine) demonstrates YOLOv3 at 220 FPS for screen-based game control; multiple Human Benchmark bots on GitHub demonstrate the exact MSS + pyautogui pattern.

---

## Conclusion: the convergence is real but incomplete

Three developments make this an inflection point. First, **GUI-VLA models exist and work** — ShowUI, UI-TARS, and SeeClick prove that the vision → reason → act pattern transfers from robot arms to computer screens. Second, **high-frequency VLA control is solved for robotics** — streaming inference achieves 480 Hz on consumer GPUs, and action chunking decouples inference speed from control speed. Third, **sub-billion parameter models run on laptops** — SmolVLA at 450M parameters fits in 0.9 GB and achieves 18ms inference.

What's missing is the explicit synthesis: a model trained on continuous computer interaction data (mouse trajectories, keystroke timings) at human-native resolution, running at 30+ Hz, treating the screen the way FSD treats the road. Game-TARS and JARVIS-VLA come closest for games; Tesla's Digital Optimus is the largest bet on this idea for general computer use, though it remains vaporware. The empirical evidence from Game-TARS is perhaps the most important signal — unified keyboard/mouse action spaces show **superior scaling** compared to domain-specific abstractions, suggesting that the "first principles" decomposition will eventually win as data and compute grow. A researcher with a MacBook, SmolVLA, MSS, and Quartz CoreGraphics can build a working prototype of this vision today.