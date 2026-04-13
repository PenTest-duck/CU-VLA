# Experiment 3: MiniWoB-Pygame — Computer Use Primitive Task Suite

## Motivation

MiniWoB++ (https://miniwob.farama.org) contains ~100 web interaction tasks that test a broad range of computer use primitives. However, it runs in Selenium/Chrome which introduces 20-50ms step overhead, making it unsuitable for our 30Hz+ training regime. Additionally, its action space uses absolute coordinates and high-level actions (CLICK_ELEMENT, TYPE_TEXT) that don't match our event-level primitive philosophy.

**Decision: Recreate MiniWoB++ task patterns in Pygame**, keeping our fast 1000+ FPS training capability and native (dx, dy, button_state, key_class) action space, while preserving the task diversity and progressive difficulty that makes MiniWoB++ valuable as a benchmark design.

When we reach VLA model stage, we may additionally build a thin MiniWoB++ adapter for standardized evaluation, but training always happens in Pygame.

## Unified Action Space

All tasks share a single action format, extending naturally from Experiments 1 and 2:

```python
action = {
    "dx": float,          # cursor delta x, clipped to [-max_delta_px, max_delta_px]
    "dy": float,          # cursor delta y, clipped to [-max_delta_px, max_delta_px]
    "button": int,        # 0=no_change, 1=mouse_down, 2=mouse_up
    "key": int,           # 0=no_key, 1-26=A-Z, 27=space, 28=enter, 29=backspace, 30=tab, 31-40=0-9
}
```

**Changes from Exp 2:** Added enter (28), backspace (29), tab (30) for form navigation, and digits 0-9 (31-40) for numeric entry. Total: 41 key classes. All letters are uppercase; lowercase/mixed-case deferred to VLA stage when Shift modifier is added. Modifier keys (shift, ctrl) can be added later as a bitfield if needed.

**Step semantics:** At each 30Hz step, the environment applies actions in canonical order: (1) move cursor by (dx, dy), (2) apply button state change, (3) apply key press. Simultaneous mouse+key in one step is allowed (e.g., moving while typing), matching how real input events can overlap in time.

## Unified Observation Space

```python
observation = {
    "screenshot": np.ndarray,   # (obs_size, obs_size, 3) uint8 RGB
    "cursor_pos": np.ndarray,   # (2,) float32, normalized [0, 1] x [0, 1]
}
```

- `obs_size` = 224 (same as Exp 2, compatible with vision backbones)
- `window_size` = 400 (same as Exp 1+2)
- Task instruction is rendered as text in the top portion of the Pygame window (like MiniWoB++ header bar), so the vision backbone sees it as pixels
- When we move to VLA stage, instruction will be provided as a separate text token input

## Task Instruction Encoding (Phased)

- **ACT stage (now):** Instruction rendered as text pixels in the frame header (e.g., "Drag the RED shape to the RED zone"). The vision backbone must learn to read this.
- **VLA stage (future):** Instruction provided as a separate text input to a language encoder. The rendered text may still be present in the frame for redundancy.

## Task Suite — Phased Roadmap

### Phase 1: Core Primitives (4 tasks)
Validate the 4 fundamental motor primitives individually. Minimal language grounding needed.

#### 1a. click-target
- **Inspired by:** MiniWoB++ click-test, click-test-2, click-color
- **Description:** A colored shape appears at a random position. Agent must move cursor to it and click.
- **Instruction (rendered):** "Click the {color} {shape}" or none (single target, obvious)
- **Motor primitive:** Point + click
- **Success criterion:** Click within shape bounding box
- **Variants:** Single target, multiple distractors (click the right color), moving target
- **Status:** Essentially solved by Exp 1 (96.5% hit rate). Include as regression baseline.

#### 1b. drag-to-zone
- **Inspired by:** MiniWoB++ drag-box, drag-items, drag-shapes
- **Description:** Colored shapes on left, matching drop zones on right. Drag shape to zone.
- **Instruction (rendered):** "Drag the {color} shape to the {color} zone"
- **Motor primitive:** Sustained mousedown + trajectory + mouseup
- **Success criterion:** Shape center within zone bounds on release
- **Variants:** 1-3 shapes, varying distances, obstacle avoidance
- **Status:** This IS Experiment 2 (drag-and-label). Include with typing disabled for pure drag eval.

#### 1c. use-slider
- **Inspired by:** MiniWoB++ use-slider, use-slider-2
- **Description:** Horizontal slider with a target value displayed. Agent must drag handle to target.
- **Instruction (rendered):** "Set the slider to {value}" (value shown as number + tick mark)
- **Motor primitive:** Fine-grained 1D continuous drag
- **Success criterion:** Slider value within ±tolerance of target
- **Variants:** Horizontal/vertical, different ranges, logarithmic scale
- **Why valuable:** Tests fine motor precision. Unlike drag-to-zone (hit a big target), this requires stopping at a specific position along a continuous axis.

#### 1d. type-field
- **Inspired by:** MiniWoB++ enter-text, enter-password
- **Description:** Text input field with a target string displayed. Agent must click the field and type the string.
- **Instruction (rendered):** "Type: {word}" (word displayed above input field)
- **Motor primitive:** Click to focus + sequential key presses
- **Success criterion:** Field value matches target string
- **Variants:** Short words (3 chars, like Exp 2), longer strings, mixed case, numbers
- **Why valuable:** Isolates typing from dragging. Tests whether the model can plan keystroke sequences from visual instruction.

### Phase 2: Compositions & Precision (4 tasks)
Combine primitives and test precision control.

#### 2a. click-sequence
- **Inspired by:** MiniWoB++ click-button-sequence, ascending-numbers
- **Description:** Multiple numbered/labeled buttons. Agent must click them in specified order.
- **Instruction (rendered):** "Click in order: 3, 1, 4, 2" or numbered labels visible
- **Motor primitive:** Sequential point + click with ordering constraint
- **Why valuable:** Tests working memory / sequential planning in the action chunking framework. The model must plan a trajectory visiting multiple targets.

#### 2b. draw-path
- **Inspired by:** MiniWoB++ draw-line, draw-circle
- **Description:** Two points (or a shape outline) displayed. Agent must draw a line/circle by holding mouse down and tracing the path.
- **Instruction (rendered):** "Draw a line from A to B" or "Trace the circle"
- **Motor primitive:** Continuous freehand trajectory with mouse held down
- **Why valuable:** Tests smooth path generation. Unlike drag-to-zone (point A to point B), this requires following a specific curved path. Directly tests action chunking's ability to produce coherent trajectories.

#### 2c. highlight-text
- **Inspired by:** MiniWoB++ highlight-text, highlight-text-2
- **Description:** A paragraph of text rendered. Agent must click-drag to highlight a specific word or phrase.
- **Instruction (rendered):** "Highlight the word '{word}'"
- **Motor primitive:** Precision drag with exact start and end positions
- **Why valuable:** Combines visual text recognition with precision mouse control. Requires finding the word, positioning cursor at its start, mousedown, dragging to end, mouseup. Very common real-world computer use operation.

#### 2d. drag-sort
- **Inspired by:** MiniWoB++ drag-sort-numbers
- **Description:** Numbered cards in random order. Agent must drag them into ascending order.
- **Instruction (rendered):** "Sort the numbers in ascending order"
- **Motor primitive:** Multiple sequential drag operations with planning
- **Why valuable:** Tests multi-step planning. The model must decide WHICH card to move first and WHERE. Requires reasoning about ordering, not just visuomotor execution.

### Phase 3: Multi-Primitive Tasks (4 tasks)
Tasks requiring multiple primitive types within a single episode.

#### 3a. form-fill
- **Inspired by:** MiniWoB++ login-user, multi-orderings, form-sequence
- **Description:** Form with 2-3 fields (username, password, dropdown). Agent must fill and submit.
- **Instruction (rendered):** "Username: alice, Password: x7k" (+ submit button)
- **Motor primitives:** Click to focus + type + tab between fields + click submit
- **Why valuable:** Tests primitive composition. Real computer use is almost always multi-primitive. This is the simplest multi-step workflow.

#### 3b. drag-and-label (Exp 2 extended)
- **Inspired by:** Our own Exp 2 + MiniWoB++ drag-items
- **Description:** Drag shapes to zones, then type their labels. This IS Exp 2 but framed within the unified task suite.
- **Motor primitives:** Drag + type
- **Why valuable:** Already have this. Including it validates backward compatibility.

#### 3c. scroll-and-click
- **Inspired by:** MiniWoB++ click-scroll-list, scroll-text
- **Description:** A scrollable list taller than the viewport. Target item is off-screen. Agent must scroll to find it, then click.
- **Instruction (rendered):** "Click on '{item_name}'"
- **Motor primitives:** Scroll + visual search + click
- **Extension to action space:** Requires scroll action. Options: (a) add scroll_direction to action dict, (b) implement scroll as sustained mouse drag on a scrollbar widget.
- **Why valuable:** Introduces viewport management, a fundamental computer use capability.

#### 3d. copy-paste
- **Inspired by:** MiniWoB++ copy-paste
- **Description:** Text displayed in one region. Empty text field in another. Agent must highlight source text, copy (Ctrl+C), click target field, paste (Ctrl+V).
- **Instruction (rendered):** "Copy the text below and paste it into the field"
- **Motor primitives:** Highlight (precision drag) + keyboard shortcut + click + paste shortcut
- **Extension to action space:** Requires modifier keys (Ctrl). Options: (a) add modifier bitfield to action, (b) add Ctrl+C and Ctrl+V as special key classes.
- **Why valuable:** Tests the full loop of reading, selecting, and reproducing text via keyboard shortcuts. Very common real-world operation.

### Phase 4: Advanced / Language-Heavy (Future, VLA stage)
These require genuine language understanding, saved for when we add a language encoder.

| Task | Inspired by | Why defer |
|------|-------------|-----------|
| **search-and-click** | MiniWoB++ search-engine | Requires typing a query, reading results, clicking correct link |
| **email-compose** | MiniWoB++ email-inbox-reply | Multi-step: navigate inbox, select email, compose reply, send |
| **calendar-event** | MiniWoB++ daily-calendar, choose-date | Date picker interaction, time entry, form fill |
| **multi-tab-navigate** | MiniWoB++ click-tab, navigate-tree | Hierarchical navigation with backtracking |

## Metrics Framework

All tasks report a common set of metrics for cross-task comparison:

### Universal Metrics
- **Success rate:** Binary task completion (%)
- **Completion time:** Steps to completion (at 30Hz, convert to seconds)
- **Control frequency:** Actual achieved Hz (should be ≥30)

### Per-Primitive Metrics
| Primitive | Metrics |
|-----------|---------|
| **Click** | Hit rate (%), reaction time (steps), distance error (px) |
| **Drag** | Drop accuracy (%), path smoothness (jerk), drag-hold consistency (% frames with correct button state) |
| **Type** | Character accuracy (%), word accuracy (%), keystrokes per char |
| **Scroll** | Target found rate (%), scroll overshoot |
| **Draw** | Path IoU with target shape, Fréchet distance to reference path |

### Aggregate Scores
- **Motor score:** Weighted average across primitive metrics
- **Composition score:** How well performance holds when primitives are combined (Phase 3 vs Phase 1 deltas)

## Base Environment Class

All tasks extend a shared `BaseTaskEnv`:

```python
class BaseTaskEnv:
    """Shared Pygame environment for all MiniWoB-Pygame tasks."""

    # Shared config
    window_size: int = 400
    obs_size: int = 224
    control_hz: int = 30
    max_delta_px: float = 50.0
    instruction_bar_height: int = 40  # top bar for rendered instruction text

    # Common methods
    def reset(self) -> tuple[dict, dict]: ...
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]: ...
    def render_instruction(self, text: str): ...  # renders text in top bar
    def get_observation(self) -> dict: ...  # screenshot + cursor_pos
    def apply_cursor_delta(self, dx, dy): ...  # clamp to window bounds
    def apply_button(self, btn): ...  # track mouse state
    def apply_key(self, key): ...  # dispatch to focused widget
```

## Connection to Existing Experiments

| Experiment | Relationship to Exp 3 |
|------------|----------------------|
| **Exp 1 (Reactive Clicks)** | click-target task is a superset. Exp 1 model can be evaluated on it directly. |
| **Exp 2 (ACT Drag-and-Label)** | drag-and-label task IS Exp 2, reframed. Exp 2 model can be evaluated on drag-to-zone and type-field individually. |
| **Exp 3 (MiniWoB-Pygame)** | Unified framework. New models trained here should generalize across all tasks. |

## Multi-Task Training Strategy

With a unified action space, we can train a single model on multiple tasks:

1. **Single-task:** Train + eval on each task independently (baseline)
2. **Multi-task BC:** Mix expert demonstrations from all tasks. Model must generalize.
3. **Task-conditioned:** Model receives task ID or instruction embedding as additional input. Can specialize while sharing the visuomotor backbone.
4. **Curriculum:** Start with Phase 1 tasks, progressively add harder ones.

## Delta-to-Absolute Adapter (for future MiniWoB++ evaluation)

If/when we want to evaluate on real MiniWoB++:

```python
class DeltaToAbsoluteWrapper(gymnasium.Wrapper):
    """Converts our (dx, dy, button, key) actions to MiniWoB++ absolute actions."""

    def __init__(self, env, max_delta_px=50.0):
        super().__init__(env)
        self.cursor_x = env.unwrapped.screen_width / 2
        self.cursor_y = env.unwrapped.screen_height / 2
        self.max_delta_px = max_delta_px
        self.mouse_pressed = False

    def step(self, action):
        # Convert delta to absolute
        self.cursor_x += action["dx"] * self.max_delta_px
        self.cursor_y += action["dy"] * self.max_delta_px
        self.cursor_x = np.clip(self.cursor_x, 0, self.screen_width)
        self.cursor_y = np.clip(self.cursor_y, 0, self.screen_height)

        # Convert button state to MiniWoB++ action type
        if action["button"] == 1:  # mouse_down
            mwob_action = create_action(MOUSEDOWN_COORDS, coords=[self.cursor_x, self.cursor_y])
            self.mouse_pressed = True
        elif action["button"] == 2:  # mouse_up
            mwob_action = create_action(MOUSEUP_COORDS, coords=[self.cursor_x, self.cursor_y])
            self.mouse_pressed = False
        else:
            mwob_action = create_action(MOVE_COORDS, coords=[self.cursor_x, self.cursor_y])

        return self.env.step(mwob_action)
```

## Decisions Made

- **Reward:** Sparse terminal reward by default (+1 success, -1 fail), matching MiniWoB++. Dense shaped reward available as an opt-in flag for future RL experiments. BC from expert demos doesn't use reward.
- **Instruction rendering:** Large bold text (24-28px), white on dark gray, in a 40px header bar at top of screen. High contrast for reliable reading by vision backbones.
- **Key scope:** A-Z uppercase + digits 0-9 + space/enter/backspace/tab = 41 classes. Lowercase and punctuation deferred to VLA stage (requires Shift modifier).
- **Widget tasks (checkbox, dropdown, etc.):** Not in scope. Our research thesis is about visuomotor control frequency, not widget coverage.
- **Phase ordering:** Provisional. Phase labels represent intended complexity progression but may be reordered based on pilot results. Reviewer noted highlight-text (Phase 2) may be harder than some Phase 3 tasks.

## Open Questions

1. **Observation frame rate vs action frame rate:** Should the model see every frame at 30Hz, or can we decouple (e.g., observe at 10Hz, act at 30Hz with chunked actions)?
2. **Task difficulty curriculum:** Within each task, how do we scale difficulty? (e.g., number of distractors, target size, time limit)
3. **Cross-task transfer:** Will a ResNet18/DINOv2 backbone trained on click-target transfer to drag-to-zone without fine-tuning? This is a key experiment.
4. **Visual domain gap:** Pygame rendering differs significantly from real browser rendering (fonts, anti-aliasing, CSS layout). If we later evaluate on real MiniWoB++, transfer may be weak. Consider a "browser skin" rendering mode for Pygame.
5. **Multi-task negative transfer:** Mixing easy tasks (click-target) with hard tasks (highlight-text) may cause gradient domination. Curriculum learning is not optional detail -- it's load-bearing for multi-task training.

## Reviewer Feedback (Incorporated)

An independent reviewer flagged the following (addressed inline above):
- Keyboard scope was inconsistent with stated task variants → fixed (added digits)
- Step semantics for simultaneous mouse+key were undefined → added canonical ordering
- Phase 3 bundles too many action-space extensions → acknowledged, phases are provisional
- Missing widget primitives → out of scope per research focus
- highlight-text may be harder than its phase suggests → noted in Decisions
- dx/dy units must be consistent between env, wrapper, and training → TODO: audit when implementing
