# Experiment 5: Mini Text Editor — VLA Task Design

## Motivation

Experiments 1-3 validated visuomotor control (V+A) on Pygame tasks with increasing complexity. However, none truly tests language grounding — the "L" in VLA. The model always does the same thing regardless of instruction text.

**This experiment introduces a task where the same visual scene requires different action sequences depending on a natural language instruction.** A mini text editor provides a natural, computer-use-relevant setting: the model must read the editor state (V), parse an edit instruction (L), and execute a multi-step motor sequence to carry out the edit (A).

## Task Description

A Pygame-rendered text editor. Each episode starts with a randomized text snippet and a randomly sampled instruction. The model must execute the edit using low-level mouse + keyboard primitives at 30 Hz.

### 4 Operations

| Op | Example instruction | Action sequence |
|----|-------------------|-----------------|
| Click-to-position | "Click after the word 'fox'" | move mouse to word end → click |
| Click + type | "Click after 'fox' and type 'brown'" | move → click → type keys one per frame |
| Select + delete | "Select the word 'hello' and delete it" | move → click word start → shift+click word end → Delete |
| Replace | "Replace 'quick' with 'slow'" | move → click word start → shift+click word end → type new word |

### Why This Task

- **V**: Must visually locate words, cursor position, and text boundaries in rendered editor
- **L**: Must parse instruction to determine *which* operation on *which* target — same visual scene, different instruction = completely different action sequence
- **A**: Multi-step motor sequences: navigate cursor → click → (shift+click to select) → type/delete. Tests action chunking on variable-length trajectories
- **Scriptable expert**: Expert knows exact pixel positions from font renderer, no model needed
- **Non-trivial**: Select+delete requires coordinated shift+click sequences (4-6 sequential sub-actions)

## Environment: `MiniEditorEnv`

### Window Layout

- **Total window**: 640×480 Pygame (wider aspect ratio suits a text editor)
- **Observation**: Rendered to 512×384 RGB (or nearest backbone-compatible size)
- **Layout**: Minimal — white background, thin border, monospace text with small margin. No line numbers, no status bar, no menu, no instruction bar. Just text.
- **No scrolling**: Text always fits entirely on screen. Passages are selected/truncated to fit within the visible area.

```
┌─────────────────────────────────────────┐
│                                         │
│  Tom's order (#47) cost $12.95;         │
│  he paid with a Visa card.              │
│                                         │
│                    ↖                     │  ← mouse cursor
│                                         │
│                                         │
└─────────────────────────────────────────┘
         640 × 480 pixels
```

**Instruction is NOT rendered on screen.** It is provided as a separate text input to the model's language encoder. The model sees only the editor contents visually.

**Resolution rationale**: At 640×480 with a 20px monospace font, each character is ~12px wide. After downsampling to 512×384, characters are ~10px wide — sufficient for visual distinction. The 224×224 resolution from Exp 1-3 is too small for text (~6px/char after downsample, below legibility threshold).

### Text Corpus

**Source**: [`agentlans/high-quality-english-sentences`](https://huggingface.co/datasets/agentlans/high-quality-english-sentences) from HuggingFace.

**Filtering**: Keep only sentences where every character is typeable with our 53-key action space. The typeable character set is the 95 printable ASCII characters (0x20–0x7E): `a-z A-Z 0-9` plus `` ` ~ ! @ # $ % ^ & * ( ) - _ = + [ ] { } \ | ; : ' " , < . > / ? `` plus space. Reject any sentence containing accented characters, emoji, non-ASCII unicode, em-dashes (—), curly quotes, etc.

**Additional filters**:
- Length: keep sentences that fit within the editor area when word-wrapped (~30-32 chars/line × ~8-10 visible lines at 20px monospace in 640×480). Roughly 60-250 characters per passage.
- Combine 1-3 sentences per episode to get multi-line text blocks.
- **All targetable words are unique within an episode** — no duplicate target words, eliminating instruction ambiguity. Common function words like "the", "a", "is" may repeat but are never selected as instruction targets.
- Words used as instruction targets are ≥3 characters (avoids tiny click targets like "a" or "to").
- Reject passages with fewer than 4 targetable words (need enough targets to sample from).

### Cursor Model (Separate, Like Real OS)

Two cursors:
- **Mouse cursor**: Pixel position `(x, y)`, moves with `(dx, dy)` deltas. Drawn as arrow/crosshair.
- **Text cursor**: Character position `(line, char_index)`, set by clicking. Drawn as blinking `|` between characters.

Click flow: mouse at pixel `(152, 200)` → click → find nearest character boundary → text cursor = `(line=0, char=7)`.

### Step Mechanics (30 Hz) — Physical Keyboard Model

The env models a physical Mac keyboard exactly. Each frame, the agent provides the binary held-state of all 53 keys + mouse. The env compares against the previous frame to detect transitions and process them as a real OS would.

**Per-frame processing order:**

1. **Apply mouse delta** `(dx, dy)` — mouse cursor moves in pixel space
2. **Detect mouse_left transition:**
   - 0→1 (mouseDown): check if either shift key is currently held (state=1 this frame)
     - No shift: set text cursor at nearest character boundary to mouse position
     - Shift held: extend selection from current text cursor to click position
   - 1→0 (mouseUp): no action (drag release)
3. **Detect key transitions** by diffing `keys_held[t]` vs `keys_held[t-1]`:
   - For each key that transitioned 0→1 (keyDown):
     - **Modifier keys** (LShift, RShift): no character generated, just update modifier state
     - **Character keys** (letters, numbers, symbols): check if either shift key is held (state=1 this frame) → generate the appropriate character (shifted or unshifted) → insert at text cursor / replace selection
     - **Delete**: delete selection if one exists, otherwise delete character before text cursor
     - **Return**: insert newline at text cursor
     - **Space**: insert space at text cursor
     - **Tab**: insert tab at text cursor
   - For each key that transitioned 1→0 (keyUp): no character generated
   - **Key repeat**: disabled. A held key only generates one character on the initial 0→1 transition. Holding a key across multiple frames has no further effect until it is released and pressed again.

**Key principle:** The env processes **transitions** (edges), not states. The held-state action space is the physical ground truth; the env derives events from state changes. This matches how a real OS keyboard driver works.

**Simultaneous key presses:** USB HID keyboards report held keys as an unordered set per poll cycle. When multiple keys transition 0→1 in the same frame, there is physically no ordering information. The env mirrors this reality:

- **Modifier + character key** (e.g. LShift + `4`): modifiers are processed first (they are a separate bitmask in USB HID reports), then character keys. So shift is "held" when `4`'s keyDown fires → generates `$`. This matches real USB HID report structure where modifiers are a dedicated byte.
- **Multiple character keys** (e.g. `A` + `S` both 0→1): processed in key index order (our analog of HID usage code order). Both characters are inserted. This is rare in practice — the expert never produces it, and at 30Hz a human almost never would either. But the env handles it deterministically rather than treating it as an error.

### Selection Highlight

When text is selected (via shift+click), the selected range is rendered with a visible blue highlight background — identical to how real text editors display selections. This is critical for learnability: the model does not need to "remember" that a selection is in progress across action chunks. The current frame visually shows the highlight, and the proprio state encodes `shift=held, mouse_left=held`. The visuomotor mapping at each re-planning step is unambiguous: "highlight visible + shift held + instruction says 'select X' → continue moving toward end of word."

### Episode Structure

- **Setup**: Random text + random instruction sampled
- **Max steps**: 300 (~10s at 30 Hz). Longer than Exp 1-3 to accommodate multi-step operations (select+delete can take 80-150 steps, plus margin).
- **Termination**: Early termination when the edit is complete (editor text matches expected result). This avoids filling the dataset with idle frames that would bias the policy toward inaction.
- **Initial state**: Mouse cursor at center of editor area. No text cursor set (blinking `|` not visible until first click).

### Type-at-Cursor Operation

The "type at cursor" operation is always preceded by a click-to-position within the same episode. The instruction is compound: "Click after '{word}' and type '{text}'". This eliminates the ambiguity of a "pre-positioned" cursor — every episode starts from scratch, and the model always performs the full click → type sequence. This also makes the type operation more interesting: it tests cursor navigation *and* typing in one episode.

### Instruction Sampling

Uniform random across 4 operation types. Target word sampled randomly from current text. For "type" operations, new text sampled from vocabulary. For "replace", both target and replacement sampled.

Instruction templates (~50-100 unique phrasings):
- "Click after the word '{word}'"
- "Position the cursor after '{word}'"
- "Click after '{word}' and type '{text}'"
- "Place cursor after '{word}' and insert '{text}'"
- "Select the word '{word}' and delete it"
- "Delete the word '{word}'"
- "Replace '{word}' with '{new_word}'"
- "Change '{word}' to '{new_word}'"

## Action Space

Multi-binary held-state format, with keys mapped to **physical Mac keyboard keys** (not logical characters). The environment interprets shift+key combos into the correct character.

```python
action = {
    "dx": float,           # cursor delta → 49 discrete exponential bins
    "dy": float,           # cursor delta → 49 discrete exponential bins
    "mouse_left": int,     # 0=released, 1=held (binary held state)
    "keys_held": [53],     # binary per physical key:
                           #   0-25:  A-Z (letter keys)
                           #   26-35: 0-9 (number row; shifted → ) ! @ # $ % ^ & * ( )
                           #   36:    backtick `    (shifted → ~)
                           #   37:    minus -       (shifted → _)
                           #   38:    equals =      (shifted → +)
                           #   39:    left bracket [  (shifted → {)
                           #   40:    right bracket ] (shifted → })
                           #   41:    backslash \   (shifted → |)
                           #   42:    semicolon ;   (shifted → :)
                           #   43:    apostrophe '  (shifted → ")
                           #   44:    comma ,       (shifted → <)
                           #   45:    period .      (shifted → >)
                           #   46:    slash /       (shifted → ?)
                           #   47:    LShift
                           #   48:    RShift
                           #   49:    Space
                           #   50:    Delete
                           #   51:    Return
                           #   52:    Tab
}
```

**53 physical keys** — the minimal set covering all characters in typical English text (letters, digits, punctuation, symbols).

**Shift behavior**: LShift and RShift are separate physical keys. The expert uses proper opposite-hand shift convention (e.g., RShift + A for capital 'A', LShift + period for '>'). The env checks if either shift key is held when interpreting a key press.

**Shift+click for selection** = `mouse_left=1, keys_held[47]=1` (or `keys_held[48]=1`) simultaneously.

**Relation to Exp 3**: Exp 3 used 43 keys. This expands to 53 by adding the 11 symbol keys and splitting shift into LShift/RShift (replacing the single shift + ctrl + alt). Ctrl and Alt are dropped — not needed for this task.

## Observation Space

```python
observation = {
    "screenshot": np.ndarray,    # (512, 384, 3) uint8 RGB
    "proprio": np.ndarray,       # (56,) float32:
                                 #   [0-1]  cursor_x, cursor_y (normalized [0,1])
                                 #   [2]    mouse_left (0 or 1)
                                 #   [3-55] keys_held (53 binary, current physical key state)
    "instruction": str,          # natural language instruction text
}
```

**Proprio rationale**: The full 56-dim proprioception vector gives the model awareness of all currently held keys and mouse state. This is critical for selection operations — the model needs to know shift is held and mouse is down without relying solely on the visual highlight. Matches how a human has proprioceptive awareness of their fingers on the keyboard.

The `instruction` field is new vs Experiment 3. It is NOT rendered in the screenshot — the model must process it through a language encoder. This makes the task a true V+L+A test: the visual scene alone is ambiguous without the instruction.

## Expert Policy

Fully scripted — no model in the loop. The expert generates the instruction, so it knows target word pixel positions exactly via `font.size()`.

### Per-Operation Expert Flow

**1. Click-to-position**
- Compute pixel coords of right edge of `{word}` using font metrics
- Fitts's Law trajectory: current mouse pos → target pos
- Click (mouse_left 0→1→0)

**2. Click + type** (compound operation)
- Fitts's Law trajectory → click after `{word}` (sets text cursor)
- Brief pause (2-3 frames)
- Emit one key press per frame for each character in `{text}`
- Shift handling: see "Expert Shift Timing" below

**3. Select + delete**
- Fitts's Law trajectory → click at first character of word (sets text cursor)
- Brief pause (2-3 frames)
- Hold shift → Fitts's Law trajectory → click at last character of word (extends selection)
- Release shift → press Delete (deletes selection)

**4. Replace**
- Same as select+delete, but type `{new_word}` instead of just Delete
- Selection replaced by first typed character; remaining chars inserted

### Expert Shift Timing

The env accepts any timing (simultaneous or sequential) — it just processes transitions. The expert uses **2-frame sequential** timing as its convention:

- Frame N: shift down, character key still up
- Frame N+1: shift still held, character key down (env sees keyDown + shift held → shifted char)
- Frame N+2: both released (unless next char is also shifted)

For consecutive shifted chars (e.g. `ABC`), shift stays held across them:
```
[RShift=1, A=0] → [RShift=1, A=1] → [RShift=1, A=0] → [RShift=1, B=1] → [RShift=1, B=0] → [RShift=1, C=1] → [RShift=0, C=0]
```

**Opposite-hand shift convention:** RShift for left-hand keys (A-G, 1-5, etc.), LShift for right-hand keys (H-Z, 6-0, etc.). Matches proper touch-typing form.

Note: This is the expert's choice, not an env constraint. A policy that uses simultaneous `[Shift=1, A=1]` in a single frame would also work — the env processes modifiers before character keys within a frame.

### Trajectory Properties
- Base trajectories from Fitts's Law curves (reused from `experts/common.py`), augmented with the variance described above
- Variable episode length: click ~30-80 steps, click+type ~80-200, select+delete ~100-200, replace ~120-250 (longer than deterministic estimates due to pauses and corrections)

## Data Generation Pipeline

### Stage 1: Corpus Preparation

1. Load `agentlans/high-quality-english-sentences` from HF Hub
2. Filter: keep only sentences where every character is in the printable ASCII range (0x20–0x7E)
3. Filter: keep sentences of 20-120 characters
4. Combine 1-3 sentences into passages of 60-250 characters
5. Word-wrap each passage at ~30-32 chars/line (fitting 640×480 at monospace 20px)
6. Reject passages with fewer than 4 unique targetable words (≥3 chars)
7. Cache the filtered corpus for reuse across generation runs

### Stage 2: Episode Sampling

For each episode:
1. Sample a random passage from the filtered corpus
2. Pick an operation type uniformly: {click, click+type, select+delete, replace}
3. Pick a target word randomly from targetable words in the passage
4. Generate instruction text from templates (with phrasing variation)
5. For click+type and replace: sample new text (1-3 words) from corpus vocabulary
6. Compute expected final text by applying the edit programmatically
7. Compute target word pixel positions using `font.size()` per-line

### Stage 3: Expert Execution (State Machine)

The expert runs as a deterministic state machine inside the env. Each frame it outputs an action based on its current phase.

**Click-to-position:**
```
MOVING_TO_TARGET → MOUSE_DOWN (1f) → MOUSE_UP (1f) → DONE
```

**Click + type:**
```
MOVING_TO_TARGET → MOUSE_DOWN (1f) → MOUSE_UP (1f) → PAUSE (2-3f) → TYPING → DONE
```
Typing: per character, unshifted = key down (1f) → key up (1f). Shifted = shift down (1f) → shift+key down (1f) → both up (1f). Consecutive shifted chars keep shift held.

**Select + delete:**
```
MOVING_TO_WORD_START → CLICK_START (2f) → PAUSE (2-3f) → SHIFT_DOWN (1f) → MOVING_TO_WORD_END → CLICK_END (2f, shift held) → SHIFT_UP (1f) → DELETE (2f) → DONE
```

**Replace:**
Same as select+delete through CLICK_END, then TYPING phase instead of DELETE. First typed character replaces selection.

**Movement phases** use Fitts's Law trajectories (reused from `experts/common.py`) with human-like variance injected at four levels:

**1. Mouse trajectory variance:**
- **Curvature**: Random bezier control point offset perpendicular to the straight-line path (~5-15% of distance). Trajectories arc slightly rather than following a perfectly straight line.
- **Submovements**: With ~30% probability, the trajectory overshoots the target by 5-20px, pauses 2-5 frames, then corrects back. Models the human overshoot-and-correct pattern.
- **Speed profile noise**: ±10-20% per-frame velocity noise on top of the smooth Fitts's Law speed profile.
- **Micro-jitter**: 1-2px Gaussian noise per frame on both axes throughout movement.

**2. Click timing variance:**
- **Pre-click dwell**: 2-8 frames (66-266ms) after arriving at target before clicking. Sampled from log-normal distribution.
- **Click duration**: mouse_down held for 1-4 frames (33-133ms), sampled per click.
- **Click position scatter**: ±3-5px Gaussian offset from the exact target position. **Final click position is clamped to the target word's bounding box** — noise is applied freely during movement, but the click event always lands within the target. For selection, both start-click and end-click are clamped to the word's first/last character boundaries.

**3. Typing rhythm variance:**
- **Inter-key interval (IKI)**: Varies per keystroke. Common bigrams (th, er, in): 1-2 frames. Uncommon transitions: 3-5 frames. After space: 2-4 frames. After shifted char: +1-2 frames overhead.
- **Per-episode typing speed**: Overall speed multiplier sampled per episode (0.7×–1.3×) to simulate fast vs slow typists.
- **Shift timing jitter**: Shift press leads character key by 1-3 frames (not always exactly 1). Shift release lags by 0-2 frames after key release.
- **Micro-pauses**: ~5% probability per character of a 3-8 frame hesitation pause.

**4. Phase transition pauses:**
- **Post-click → typing**: 3-10 frames (log-normal).
- **Post-selection → Delete**: 2-8 frames.
- **Episode start "reading" pause**: 0-15 frames of idle before movement begins.
- **Post-arrival → click**: 1-5 frames.

### Stage 4: Recording

Each frame during expert execution, record: screenshot (JPEG q=95), action vector, cursor position, instruction text, and episode metadata. Episodes terminate early when the edit is complete.

### Stage 5: Dataset Assembly

- Assemble all episodes into a HuggingFace `Dataset`
- Save as Parquet shards, upload to HF Hub
- **Scale**: 10,000 episodes × ~100 frames avg = ~1M frames. At JPEG q=95 ≈ 30KB/frame → **~30GB total**.

## Dataset Format

Parquet via HF `datasets` library (same as Exp 2):

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | int | Unique episode identifier |
| `timestep` | int | Frame index within episode |
| `image` | Image (PNG) | 512×384 screenshot |
| `instruction` | str | Natural language instruction |
| `action_dx` | float | Mouse delta x |
| `action_dy` | float | Mouse delta y |
| `action_mouse_left` | int | Mouse button held state |
| `action_keys_held` | list[int] | 53-element binary vector (physical keys) |
| `proprio` | list[float] | 56-dim: cursor_xy(2) + mouse_left(1) + keys_held(53) |
| `operation_type` | str | click/type/select_delete/replace |
| `target_word` | str | Word being operated on |
| `expected_text` | str | Expected final editor text |
| `initial_text` | str | Starting editor text |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Task success rate** (primary) | Binary: final editor text == expected text |
| **Per-operation success** | Success rate broken down by operation type |
| **Steps to completion** | Number of steps to reach correct text (efficiency) |
| **Mouse trajectory MAE** | Mean absolute error vs expert trajectories (diagnostic) |

Normalized edit distance dropped as a primary metric — it's misleading when text is inserted at the wrong position (low edit distance but total task failure). Task success rate (exact match) is the authoritative metric.

## Model (Deferred)

The environment and expert are model-agnostic. Two model approaches will be evaluated separately:

1. **ACT + language encoder**: Extend Exp 2's ACT with a trainable text encoder. Small (~30-35M params), fully trainable on M1.
2. **SmolVLA**: Frozen VLM backbone + flow matching action expert (~450M params). Larger, riskier (frozen SigLIP on Pygame visuals).

Model architecture decisions documented separately after the environment is validated.

## Word Boundary Definition

A "word" is a maximal run of alphanumeric characters (`[a-zA-Z0-9]+`). Punctuation, symbols, and whitespace are boundaries, not part of words.

Examples:
- `Tom's` → tokens: `Tom`, `s`. Only `Tom` is targetable (≥3 chars).
- `$12.95` → tokens: `12`, `95`. Neither targetable (<3 char "words").
- `card.` → token: `card`. Period is NOT part of the word.
- `"Hello,"` → token: `Hello`. Quotes and comma excluded.
- `order` in `order (#47)` → token: `order`. Parens/hash excluded.

**Selection behavior**: "Select the word 'card'" selects exactly `c-a-r-d`, NOT adjacent punctuation or whitespace.

**Click behavior**: "Click after 'card'" places the text cursor immediately after the `d`, before any following punctuation.

**Targetable words**: alphanumeric runs ≥3 characters, unique within the episode.

## Font and Text Rendering

- **Font**: Pygame `pygame.font.SysFont("monospace", 20)` or bundled monospace TTF. Monospace simplifies pixel-to-character mapping (constant char width).
- **Char width**: ~12px at 640×480 native, ~10px after downsample to 512×384.
- **Click target**: Word-level, not character-level. "Click after 'fox'" means anywhere in the right half of the word "fox" counts. The text cursor snaps to the word boundary. This gives ~30-60px click targets (3-5 char words), comfortable for the model.
- **Line layout**: Text rendered line-by-line with word wrapping. Word positions computed per-line using `font.size(text_up_to_word)`. No word wrapping mid-word — if a word doesn't fit, it starts on the next line.
- **Text cursor**: Thin vertical bar `|` rendered between characters at the text cursor position. Blinks at 2Hz (~15 frames per cycle, visible ~50% of frames). The proprio vector always contains the ground-truth cursor position regardless of blink state, so the model is never blind to cursor location.
- **Selection rendering**: Selected text drawn with `#3399FF` (blue) background rect behind the text surface. Visible at 512×384.
- **Mouse cursor**: Rendered as a small arrow or crosshair sprite. Always visible on screen.

## Train/Eval Split

To test genuine language grounding (not memorization):
- **Instruction templates**: 80% train, 20% held-out phrasings (e.g., train on "Delete the word '{w}'" but eval includes "Remove '{w}' from the text")
- **Vocabulary**: 80% train words, 20% held-out words (eval uses unseen words in familiar templates)
- **Cross-split**: Some eval episodes use held-out phrasings with held-out words (hardest)

## Relation to Experiment 3

This task could be added as a 13th task in Exp 3's MiniWoB suite, but we treat it as a separate experiment because:
1. It introduces the language input modality (Exp 3 tasks have fixed or minimal instructions)
2. The model architecture will change (text encoder pathway)
3. It serves as the bridge to VLA, not just another visuomotor task

The environment reuses Exp 3's action space, rendering conventions, and expert trajectory utilities.
