# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
# pyrefly: ignore [missing-import]
from PIL import Image, ImageTk
import json
import math
import time
import traceback
import threading
import queue


class GlueVisionApp:
    """
    Explicit state machine (fixes the crash / camera-still-running bug):
      MODE_LIVE    -> camera actively read & shown, no boxes, no detection
      MODE_BOXING  -> frame frozen (captured), camera NOT read at all, user draws boxes
      MODE_RESULT  -> frame frozen (captured), camera NOT read at all, showing static
                      detection overlay computed ONCE on that captured image
    Camera .read() is called ONLY when self.mode == MODE_LIVE. This guarantees that
    after Capture -> draw boxes -> Start, the camera is never touched again and all
    detection runs purely on the captured still image.

    Detection engine: contour / shape matching (replaces the earlier SIFT +
    FLANN + RANSAC homography attempt, which itself had replaced
    GeneralizedHoughGuil). Shape matching was chosen because:
      - SIFT needs rich texture/keypoints. Low-texture, reflective metal
        parts (screws, brackets, glue targets) often yield <15 keypoints
        total, which is not enough to reliably describe even ONE instance,
        let alone tell several similar instances apart in the same scene.
      - Contours, by contrast, only need a clean silhouette -- which Canny
        recovers reliably regardless of surface texture or reflections.

    Pipeline:
      1. build_template_from_box(): Canny + findContours on the cropped ROI,
         keep the single largest contour as that template's shape signature
         (assumes the box was drawn tightly around one object).
      2. detect_and_overlay(): Canny + findContours ONCE on the FULL scene
         (not just inside a box) -> every contour anywhere in the frame is a
         detection candidate. For each template, cv2.matchShapes() (Hu
         moment invariants -- invariant to translation, scale and rotation
         by construction) scores every scene contour against the template
         contour. No homography, no keypoints needed at all.
      3. Candidates are ranked by shape distance (best first). Any
         candidate with distance > MAX_SHAPE_DISTANCE is rejected as "too
         different" -- this is the sole quality floor, replacing the old
         MIN_QUALITY_SCORE's role. Up to `expected_count` of the remaining
         best candidates are accepted per template.
      4. Position = contour centroid (cv2.minAreaRect center), scale =
         object's longer side / template's reference length, angle =
         minAreaRect orientation (normalized to the longer side).
    Because every scene contour is evaluated independently against the
    template, N genuinely-present similar objects anywhere in the scene are
    all found in ONE pass -- unlike the old SIFT pipeline, which needed
    enough *leftover* keypoints after each accepted match to describe a
    further, separate instance (and typically ran out after just 1).

    Multi-instance-per-template detection:
      - Each drawn box gets a "T{i} have ? Object" spinbox in the info panel
        (default 1). This sets templates[i]["expected_count"].
      - detect_and_overlay() ranks ALL scene contours by shape distance to
        each template and accepts the best `expected_count` of them that
        clear MAX_SHAPE_DISTANCE -> caps detections per template at exactly
        what the user typed.
      - MAX_SHAPE_DISTANCE acts as the hard floor: if the best remaining
        candidate's shape distance exceeds it, the search for that template
        stops immediately, even if expected_count hasn't been reached yet.
        This deliberately favors under-detection over reporting a junk /
        wrong-shaped object.

    New features vs previous version:
      - Camera selector (auto-scans indices 0..7 at startup and via
        "Rescan" button) lets the user pick which physical camera to use.
        Switching is only allowed in MODE_LIVE to avoid ambiguity with a
        frozen frame from a different device.
      - Pause/Resume button freezes the LIVE preview (camera stops being
        read) without discarding any boxes/templates/results, distinct
        from "Back" which resets state (see below).
      - "Back" is now two-level, depending on current mode:
          * MODE_RESULT -> MODE_BOXING: returns to the SAME captured frame
            with the exact boxes/counts that were used for that detection
            run restored (button label becomes "Back to Boxes"), so the
            user can tweak a box and re-run instead of redrawing from
            scratch.
          * MODE_BOXING -> MODE_LIVE: full reset, discards the captured
            frame and boxes (button label becomes "Back to Live"). This is
            the point of no return for the current capture -- there is no
            "undo capture" beyond this.
        To close the app entirely (including before ever capturing), use
        "Exit" instead of Back.
      - Exit button + window-close both go through on_close(), which asks
        for confirmation (extra warning if a detection thread is still
        running) before releasing the camera and destroying the window.
    Optimizations / fixes vs original:
      - Detection (contour extraction + matchShapes scoring) runs in a
        background thread so the Tkinter UI never freezes on slow scenes.
      - Camera open failures no longer hard-crash the whole app (clean error dialog).
      - Canvas is only redrawn when the frame actually changed (dirty-flag), instead
        of rebuilding a PhotoImage 60+ times/sec even while frozen.
      - Defensive handling of cv2.matchShapes()/cv2.findContours() edge cases
        (empty scene, template with no valid contour, etc.).
      - Pending `after()` callbacks are cancelled on close so no callback ever
        touches a released VideoCapture.
      - Buttons are disabled while a background detection job is running to avoid
        double-submission / race conditions.
    """

    MODE_LIVE = "live"
    MODE_BOXING = "boxing"
    MODE_RESULT = "result"

    def __init__(self, root):
        self.root = root
        self.root.title("Glue Vision UI")

        self.W = 960
        self.H = 540

        self.cap = None
        self.current_cam_index = 0
        self.paused = False

        self.available_cameras = self._scan_cameras()
        default_index = self.available_cameras[0] if self.available_cameras else 0
        self._open_camera(default_index, fatal=True)

        self.mode = self.MODE_LIVE

        self.last_live_frame = None      # only updated while mode == LIVE
        self.captured_frame = None       # the still image everything else works on
        self.result_frame = None         # final overlay image (computed once)

        self.boxes = []
        self.box_counts = []      # expected object count per box, parallel to self.boxes
        self.temp_box = None
        self.drawing = False
        self.start_pt = None

        # Snapshot of the boxes/counts that were active right before "Start"
        # was pressed. Restored when "Back" is pressed from MODE_RESULT, so
        # the user returns to boxing with their previous boxes intact
        # instead of an empty canvas.
        self._last_used_boxes = []
        self._last_used_box_counts = []

        self.templates = []
        self.machine_paths = []

        self.canny_low = 30
        self.canny_high = 100

        # ---------- Contour / shape matching setup (replaces SIFT) ----------
        # Shape matching finds EVERY contour in the whole scene (via Canny +
        # findContours) and scores each one against the template's contour
        # using Hu moment invariants (cv2.matchShapes). Unlike SIFT, this
        # does not depend on having many keypoints/texture -- it only needs
        # a clean, distinguishable silhouette, which is exactly what
        # low-texture metal parts (screws, brackets, etc.) provide.
        self.MIN_CONTOUR_AREA = 60        # ignore tiny noise contours in the scene
        self.MAX_SHAPE_DISTANCE = 0.35    # cv2.matchShapes() distance ceiling (lower = stricter)

        # RETR_TREE keeps INNER contours (holes, engravings, inner edges,
        # and any deeper nested levels) alongside the outer silhouette.
        # Every nesting depth found in the image becomes its own weighted
        # level when scoring a candidate: level 0 is the outer/parent
        # contour (max weight), and each deeper level n's weight decays as
        #   weight(n) = MAX_LEVEL_WEIGHT * exp(-n / sqrt(2))
        # so the outline always dominates while progressively deeper
        # nested details contribute progressively less (but decay more
        # slowly than the old e^(-n), since n/sqrt(2) < n for n > 0).
        self.MAX_LEVEL_WEIGHT = 2.0

        self._last_detect_ms = 0.0
        self._last_debug_info = ""

        self._detect_busy = False
        self._detect_queue = queue.Queue()

        # ---------- Debounced re-detect on slider change ----------
        self.REDETECT_DEBOUNCE_MS = 1000   # wait 1s after last slider move
        self._shape_thresh_after_id = None

        self._after_id = None
        self._dirty = True          # force first paint
        self._last_display_key = None

        # ---------- Layout ----------
        self.topbar = tk.Frame(root)
        self.topbar.pack(fill="x")

        self.undo_btn = tk.Button(self.topbar, text="Undo", command=self.undo_box)
        self.undo_btn.pack(side="left", padx=5, pady=5)

        self.export_btn = tk.Button(self.topbar, text="Print Paths", command=self.print_paths)
        self.export_btn.pack(side="left", padx=5, pady=5)

        self.debug_btn = tk.Button(self.topbar, text="Show Canny Debug", command=self.show_canny_debug)
        self.debug_btn.pack(side="left", padx=5, pady=5)

        cam_lbl = tk.Label(self.topbar, text="Camera:")
        cam_lbl.pack(side="left", padx=(15, 2), pady=5)

        cam_values = [str(i) for i in self.available_cameras] if self.available_cameras else ["0"]
        self.cam_select_var = tk.StringVar(value=str(self.current_cam_index))
        self.cam_select = ttk.Combobox(self.topbar, textvariable=self.cam_select_var,
                                        values=cam_values, width=4, state="readonly")
        self.cam_select.pack(side="left", padx=2, pady=5)
        self.cam_select.bind("<<ComboboxSelected>>",
                              lambda e: self.switch_camera(int(self.cam_select_var.get())))

        self.rescan_btn = tk.Button(self.topbar, text="Rescan", command=self.rescan_cameras)
        self.rescan_btn.pack(side="left", padx=(2, 5), pady=5)

        # ---------- Max shape-distance tuner ----------
        # Lets the user relax/tighten the "junk filter" from the UI instead
        # of hardcoding it. Higher value = more lenient (accepts more
        # different-looking contours as matches); lower value = stricter
        # (only near-identical shapes are accepted).
        thresh_lbl = tk.Label(self.topbar, text="Max Shape Dist:")
        thresh_lbl.pack(side="left", padx=(15, 2), pady=5)

        self.shape_thresh_var = tk.DoubleVar(value=self.MAX_SHAPE_DISTANCE)
        self.shape_thresh_scale = tk.Scale(
            self.topbar, from_=0.05, to=2.0, resolution=0.05, orient="horizontal",
            length=140, showvalue=False, variable=self.shape_thresh_var,
            command=self._on_shape_thresh_changed
        )
        self.shape_thresh_scale.pack(side="left", padx=2, pady=5)

        # Typed numeric entry as an alternative to dragging the slider --
        # lets the user key in an exact value (e.g. "0.42") and press Enter.
        self.shape_thresh_entry_var = tk.StringVar(value=f"{self.MAX_SHAPE_DISTANCE:.2f}")
        self.shape_thresh_entry = tk.Entry(self.topbar, textvariable=self.shape_thresh_entry_var,
                                            width=5, justify="center")
        self.shape_thresh_entry.pack(side="left", padx=(0, 5), pady=5)
        self.shape_thresh_entry.bind("<Return>", self._on_shape_thresh_entry_submit)
        self.shape_thresh_entry.bind("<FocusOut>", self._on_shape_thresh_entry_submit)

        self.mode_lbl = tk.Label(self.topbar, text="MODE: LIVE", font=("Segoe UI", 10, "bold"), fg="blue")
        self.mode_lbl.pack(side="right", padx=10)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame, width=self.W, height=self.H, bg="black")
        self.canvas.pack(side="left")

        self.info_panel = tk.Frame(self.main_frame, width=300, bg="#1e1e1e")
        self.info_panel.pack(side="left", fill="y")

        title_lbl = tk.Label(self.info_panel, text="System Status", bg="#1e1e1e",
                              fg="white", font=("Segoe UI", 12, "bold"))
        title_lbl.pack(anchor="w", padx=10, pady=(10, 5))

        self.status_vars = {}
        status_fields = [
            ("Mode", "Live"),
            ("Detect time", "0.0 ms"),
            ("Boxes drawn", "0"),
            ("Templates", "0"),
            ("Detections", "0"),
            ("Max shape dist", f"{self.MAX_SHAPE_DISTANCE:.2f}"),
        ]
        for key, default in status_fields:
            row = tk.Frame(self.info_panel, bg="#1e1e1e")
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{key}:", bg="#1e1e1e", fg="#aaaaaa",
                     font=("Segoe UI", 9), width=13, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text=default, bg="#1e1e1e", fg="#00ff88",
                                font=("Segoe UI", 9, "bold"), anchor="w")
            val_lbl.pack(side="left")
            self.status_vars[key] = val_lbl

        sep = tk.Frame(self.info_panel, bg="#444444", height=1)
        sep.pack(fill="x", padx=10, pady=8)

        tk.Label(self.info_panel, text="Debug Info", bg="#1e1e1e",
                 fg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(0, 5))
        self.debug_lbl = tk.Label(self.info_panel, text="", bg="#1e1e1e", fg="#ffcc00",
                                   font=("Segoe UI", 8), wraplength=280, justify="left", anchor="nw")
        self.debug_lbl.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(self.info_panel, text="Box Object Counts", bg="#1e1e1e",
                 fg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(0, 5))
        tk.Label(self.info_panel, text="Set how many instances of that object appear in the scene.",
                 bg="#1e1e1e", fg="#888888", font=("Segoe UI", 7), wraplength=280,
                 justify="left").pack(anchor="w", padx=10, pady=(0, 3))

        self.box_count_frame = tk.Frame(self.info_panel, bg="#1e1e1e")
        self.box_count_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.box_count_vars = {}   # idx -> IntVar, rebuilt whenever boxes change

        sep2 = tk.Frame(self.info_panel, bg="#444444", height=1)
        sep2.pack(fill="x", padx=10, pady=4)

        tk.Label(self.info_panel, text="Detected Objects", bg="#1e1e1e",
                 fg="white", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 5))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Info.Treeview",
                         background="#2b2b2b", fieldbackground="#2b2b2b",
                         foreground="white", rowheight=22, font=("Segoe UI", 9))
        style.configure("Info.Treeview.Heading",
                         background="#3a3a3a", foreground="white", font=("Segoe UI", 9, "bold"))

        columns = ("id", "x", "y", "angle", "scale", "quality")
        self.detect_table = ttk.Treeview(self.info_panel, columns=columns, show="headings",
                                          height=10, style="Info.Treeview")
        headings = {"id": "ID", "x": "X", "y": "Y", "angle": "Angle", "scale": "Scale", "quality": "Quality"}
        widths = {"id": 40, "x": 55, "y": 55, "angle": 55, "scale": 55, "quality": 55}
        for c in columns:
            self.detect_table.heading(c, text=headings[c])
            self.detect_table.column(c, width=widths[c], anchor="center")
        self.detect_table.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        self.bottombar = tk.Frame(root)
        self.bottombar.pack(fill="x")

        self.back_btn = tk.Button(self.bottombar, text="Back", width=12, command=self.back_to_live)
        self.back_btn.pack(side="left", padx=5, pady=5)

        self.start_btn = tk.Button(self.bottombar, text="Start", width=12, command=self.start_detection)
        self.start_btn.pack(side="left", padx=5, pady=5)

        self.capture_btn = tk.Button(self.bottombar, text="Capture", width=12, command=self.capture_frame)
        self.capture_btn.pack(side="left", padx=5, pady=5)

        self.pause_btn = tk.Button(self.bottombar, text="Pause", width=12, command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=5, pady=5)

        self.exit_btn = tk.Button(self.bottombar, text="Exit", width=12, bg="#cc4444", fg="white",
                                   command=self.on_close)
        self.exit_btn.pack(side="right", padx=5, pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind_all("<Control-z>", lambda e: self.undo_box())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.photo = None
        self._canvas_img_id = None

        self.update_loop()

    # ---------- Camera helpers ----------
    def _scan_cameras(self, max_index=8):
        """Probe camera indices 0..max_index-1 and return the ones that
        actually open. Each candidate is opened/closed quickly so this does
        not hold any device open longer than necessary.
        """
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
            cap.release()
        return found

    def _open_camera(self, index, fatal=False):
        """Open (or re-open) the camera at `index`. Releases any previously
        open capture first. On failure: shows an error dialog and either
        exits the app (fatal=True, used at startup) or leaves the previous
        camera/state untouched (fatal=False, used when switching cameras).
        """
        new_cap = cv2.VideoCapture(index)
        if not new_cap.isOpened():
            new_cap.release()
            messagebox.showerror("Camera error",
                                  f"Cannot open camera index {index}. "
                                  "Check the device is connected and not used by another app.")
            if fatal:
                raise SystemExit(1)
            return False

        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap is not None:
            self.cap.release()

        self.cap = new_cap
        self.current_cam_index = index
        self.last_live_frame = None
        return True

    def switch_camera(self, index):
        """Called from the camera-select dropdown. Only allowed in
        MODE_LIVE, since switching cameras mid-boxing/result would be
        confusing (the frozen image is unrelated to the new device)."""
        if index == self.current_cam_index:
            return
        if self.mode != self.MODE_LIVE:
            messagebox.showwarning("Warning", "Return to LIVE mode (Back) before switching camera.")
            self.cam_select_var.set(str(self.current_cam_index))
            return
        ok = self._open_camera(index, fatal=False)
        if ok:
            self._dirty = True
        else:
            self.cam_select_var.set(str(self.current_cam_index))

    def rescan_cameras(self):
        if self.mode != self.MODE_LIVE:
            messagebox.showwarning("Warning", "Return to LIVE mode (Back) before rescanning cameras.")
            return
        found = self._scan_cameras()
        self.available_cameras = found
        values = [str(i) for i in found] if found else ["0"]
        self.cam_select.config(values=values)
        if self.current_cam_index not in found and found:
            self.switch_camera(found[0])
            self.cam_select_var.set(str(found[0]))
        messagebox.showinfo("Camera scan", f"Found {len(found)} camera(s): {found}" if found
                             else "No camera found.")

    def _on_shape_thresh_changed(self, value):
        """Updates MAX_SHAPE_DISTANCE live from the topbar slider, keeps the
        typed entry box in sync, then schedules a DEBOUNCED re-detection: if
        the slider is dragged repeatedly, only the LAST change (after it has
        been still for REDETECT_DEBOUNCE_MS) actually triggers a re-run, so
        dragging fast does not spawn dozens of detection jobs and lag the UI.
        """
        try:
            self.MAX_SHAPE_DISTANCE = float(value)
        except (TypeError, ValueError):
            return

        self.shape_thresh_entry_var.set(f"{self.MAX_SHAPE_DISTANCE:.2f}")
        self._schedule_redetect_debounced()

    def _on_shape_thresh_entry_submit(self, event=None):
        """Handles typing an exact Max Shape Dist value into the entry box
        and pressing Enter (or clicking away). Same clamping range as the
        slider (0.05-2.0), keeps the slider in sync, then debounces a
        re-detection exactly like dragging the slider does."""
        raw = self.shape_thresh_entry_var.get().strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            self.shape_thresh_entry_var.set(f"{self.MAX_SHAPE_DISTANCE:.2f}")
            return

        val = max(0.05, min(2.0, val))
        self.MAX_SHAPE_DISTANCE = val
        self.shape_thresh_entry_var.set(f"{val:.2f}")
        self.shape_thresh_var.set(val)   # move the slider handle to match
        self._schedule_redetect_debounced()

    def _schedule_redetect_debounced(self):
        """Shared debounce logic used by BOTH the slider and the typed
        entry box: cancel any pending re-detect timer and start a fresh
        REDETECT_DEBOUNCE_MS one, so only the last change (slider drag OR
        keystroke) within that window actually triggers a re-run."""
        if self._shape_thresh_after_id is not None:
            try:
                self.root.after_cancel(self._shape_thresh_after_id)
            except Exception:
                pass
        self._shape_thresh_after_id = self.root.after(
            self.REDETECT_DEBOUNCE_MS, self._redetect_after_thresh_change
        )

    def _redetect_after_thresh_change(self):
        """Fired once, REDETECT_DEBOUNCE_MS after the slider stopped
        moving. Re-runs detection on the SAME captured frame + already-built
        templates (no need to redraw boxes) using the new MAX_SHAPE_DISTANCE.
        Only applies while a static result is on screen and no other
        detection job is already busy."""
        self._shape_thresh_after_id = None

        if self.mode != self.MODE_RESULT or self._detect_busy or self.paused:
            return
        if self.captured_frame is None or not self.templates:
            return

        self._detect_busy = True
        self._set_buttons_enabled(False)
        self._last_debug_info = "Re-detecting with new shape-distance threshold..."
        self._dirty = True

        frame_copy = self.captured_frame.copy()
        t = threading.Thread(target=self._redetect_worker, args=(frame_copy,), daemon=True)
        t.start()
        self.root.after(50, self._poll_detect_queue)

    def _redetect_worker(self, frame):
        """Like _detect_worker, but reuses self.templates as-is (already
        built from the last full detection) instead of rebuilding them from
        boxes -- boxes are cleared once MODE_RESULT is reached."""
        try:
            t0 = time.time()
            vis, dbg = self.detect_and_overlay(frame)
            detect_ms = (time.time() - t0) * 1000.0
            debug_text = f"Re-detected with Max Shape Dist={self.MAX_SHAPE_DISTANCE:.2f}\n" + dbg
            self._detect_queue.put(("ok", (vis, debug_text, detect_ms, None)))
        except Exception:
            err = traceback.format_exc()
            self._detect_queue.put(("error", err))

    # ---------- Pause / Exit ----------
    def toggle_pause(self):
        """Pause freezes the LIVE preview (camera .read() stops being called)
        without discarding boxes/templates/results, unlike Back which resets
        everything. Works in any mode; in BOXING/RESULT the image is already
        static, so Pause there just blocks accidental Back/Capture actions."""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(text="Resume", bg="#e0a020")
        else:
            self.pause_btn.config(text="Pause", bg=self.root.cget("bg"))
            self._dirty = True
        self._update_pause_lock()

    def _update_pause_lock(self):
        state_for_others = "disabled" if self.paused else "normal"
        for b in (self.back_btn, self.start_btn, self.capture_btn,
                  self.cam_select, self.rescan_btn, self.shape_thresh_scale,
                  self.shape_thresh_entry):
            try:
                b.config(state=state_for_others)
            except tk.TclError:
                pass

    # ---------- Mode transitions ----------
    def _set_mode(self, mode):
        self.mode = mode
        label_map = {
            self.MODE_LIVE: ("MODE: LIVE", "blue"),
            self.MODE_BOXING: ("MODE: DRAWING BOX (camera stopped)", "orange"),
            self.MODE_RESULT: ("MODE: STATIC RESULT (camera stopped)", "green"),
        }
        text, color = label_map[mode]
        self.mode_lbl.config(text=text, fg=color)

        # Back button label hints what it will do from the current mode:
        # RESULT -> restores previous boxes; BOXING -> discards capture
        # entirely and returns to LIVE (use Exit to quit before capturing).
        back_text_map = {
            self.MODE_LIVE: "Back",
            self.MODE_BOXING: "Back to Live",
            self.MODE_RESULT: "Back to Boxes",
        }
        try:
            self.back_btn.config(text=back_text_map[mode])
        except tk.TclError:
            pass

        self._dirty = True

    # ---------- UI actions ----------
    def capture_frame(self):
        if self.mode != self.MODE_LIVE or self.last_live_frame is None:
            return
        # Freeze the exact frame; camera will not be read again until Back is pressed.
        if self._shape_thresh_after_id is not None:
            try:
                self.root.after_cancel(self._shape_thresh_after_id)
            except Exception:
                pass
            self._shape_thresh_after_id = None
        self.captured_frame = self.last_live_frame.copy()
        self.result_frame = None
        self.boxes = []
        self.box_counts = []
        self.temp_box = None
        self.templates = []
        self.machine_paths = []
        self._last_used_boxes = []
        self._last_used_box_counts = []
        self._last_debug_info = ""
        self._rebuild_box_count_ui()
        self._set_mode(self.MODE_BOXING)

    def back_to_live(self):
        """Two-level Back, depending on current mode:
          - From MODE_RESULT: go back to MODE_BOXING on the SAME captured
            frame, restoring the exact boxes/counts that were used for that
            detection run (so the user can tweak a box and re-run instead
            of redrawing everything from scratch).
          - From MODE_BOXING: go all the way back to MODE_LIVE (discards
            the captured frame and any boxes -- equivalent to the old
            single-step Back). To leave the app entirely before ever
            capturing, use "Exit" instead.
        """
        if self._detect_busy:
            return

        if self._shape_thresh_after_id is not None:
            try:
                self.root.after_cancel(self._shape_thresh_after_id)
            except Exception:
                pass
            self._shape_thresh_after_id = None

        if self.mode == self.MODE_RESULT:
            # Step 1: RESULT -> BOXING, restoring the previous boxes.
            self.result_frame = None
            self.boxes = list(self._last_used_boxes)
            self.box_counts = list(self._last_used_box_counts)
            self.temp_box = None
            self.templates = []
            self.machine_paths = []
            self._last_debug_info = ""
            self._rebuild_box_count_ui()
            self._set_mode(self.MODE_BOXING)
            return

        # Step 2 (from MODE_BOXING, or as a fallback from any other mode):
        # BOXING -> LIVE, full reset. Capturing a NEW frame afterwards
        # clears _last_used_boxes naturally via start_detection's own
        # snapshot on the next run.
        self.captured_frame = None
        self.result_frame = None
        self.boxes = []
        self.box_counts = []
        self.temp_box = None
        self.templates = []
        self.machine_paths = []
        self._last_used_boxes = []
        self._last_used_box_counts = []
        self._last_debug_info = ""
        self._rebuild_box_count_ui()
        self._set_mode(self.MODE_LIVE)

    def undo_box(self):
        if self.mode == self.MODE_BOXING and self.boxes:
            self.boxes.pop()
            if self.box_counts:
                self.box_counts.pop()
            self._rebuild_box_count_ui()
            self._dirty = True

    def print_paths(self):
        print(json.dumps(self.machine_paths, indent=2))

    # ---------- Per-box expected object count UI ----------
    def _rebuild_box_count_ui(self):
        """Rebuild the small 'T{i} have ? Object' spinbox row for every
        current box. Called whenever boxes are added/removed/reset."""
        for child in self.box_count_frame.winfo_children():
            child.destroy()
        self.box_count_vars = {}

        for i in range(len(self.boxes)):
            row = tk.Frame(self.box_count_frame, bg="#1e1e1e")
            row.pack(fill="x", pady=1)

            tk.Label(row, text=f"T{i} have", bg="#1e1e1e", fg="#cccccc",
                     font=("Segoe UI", 8)).pack(side="left")

            count = self.box_counts[i] if i < len(self.box_counts) else 1
            var = tk.IntVar(value=count)
            spin = tk.Spinbox(row, from_=1, to=50, width=4, textvariable=var,
                               command=lambda idx=i: self._on_box_count_changed(idx))
            spin.pack(side="left", padx=4)
            spin.bind("<FocusOut>", lambda e, idx=i: self._on_box_count_changed(idx))
            spin.bind("<Return>", lambda e, idx=i: self._on_box_count_changed(idx))

            tk.Label(row, text="Object", bg="#1e1e1e", fg="#cccccc",
                     font=("Segoe UI", 8)).pack(side="left")

            self.box_count_vars[i] = var

    def _on_box_count_changed(self, idx):
        if idx not in self.box_count_vars:
            return
        try:
            val = int(self.box_count_vars[idx].get())
        except (tk.TclError, ValueError):
            val = 1
        val = max(1, min(50, val))
        while len(self.box_counts) <= idx:
            self.box_counts.append(1)
        self.box_counts[idx] = val

    def show_canny_debug(self):
        """Opens a live-updating Canny preview window with sliders for
        canny_low / canny_high, so the user can raise the gradient
        thresholds directly to suppress weak edges like soft shadows,
        without editing code. Any change here also affects the REAL
        detection pipeline immediately (self.canny_low/self.canny_high are
        shared with build_template_from_box / detect_and_overlay)."""
        src = self.captured_frame if self.captured_frame is not None else self.last_live_frame
        if src is None:
            messagebox.showwarning("Warning", "No image available.")
            return
        try:
            self._show_canny_tuner_window(src)
        except Exception as e:
            messagebox.showerror("Debug error", str(e))

    def _show_canny_tuner_window(self, src_bgr):
        # Use a Tkinter Toplevel instead of cv2.imshow to avoid mixing HighGUI
        # event loop with Tkinter mainloop (a known cause of crashes/freezes).
        win = tk.Toplevel(self.root)
        win.title("Canny Debug (adjustable)")

        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        img_label = tk.Label(win)
        img_label.pack()
        img_label._photo_ref = None  # keep a live reference to avoid GC

        controls = tk.Frame(win)
        controls.pack(fill="x", padx=8, pady=6)

        low_var = tk.IntVar(value=self.canny_low)
        high_var = tk.IntVar(value=self.canny_high)
        show_contours_var = tk.BooleanVar(value=True)

        def redraw(*_):
            lo = low_var.get()
            hi = high_var.get()
            if lo >= hi:
                hi = lo + 1
                high_var.set(hi)
            self.canny_low = lo
            self.canny_high = hi
            edges = cv2.Canny(blur, lo, hi)
            rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            if show_contours_var.get():
                # Overlay every cv2.findContours() point on top of the Canny
                # edges, using the SAME RETR_TREE + MIN_CONTOUR_AREA filter
                # as the real detection pipeline -- so what you see here is
                # what will actually be matched against templates.
                # Color-coded by nesting DEPTH (however many levels the
                # image actually has), matching the same weight(n) =
                # MAX_LEVEL_WEIGHT * exp(-n/sqrt(2)) scheme used in
                # detection: level 0 (outline) = green, then cycling through
                # yellow/orange/red for progressively deeper (progressively
                # lower-weight) nested levels.
                dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                raw_c, hier = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                level_colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 0, 255)]
                if raw_c and hier is not None:
                    hier = hier[0]
                    top_idx = [i for i, h4 in enumerate(hier)
                               if h4[3] == -1 and cv2.contourArea(raw_c[i]) >= self.MIN_CONTOUR_AREA]
                    for pi in top_idx:
                        lvl_map = self._build_level_map(raw_c, hier, pi)
                        for lvl, conts in lvl_map.items():
                            color = level_colors[min(lvl, len(level_colors) - 1)]
                            for c in conts:
                                for pt in c.reshape(-1, 2):
                                    cv2.circle(rgb, (int(pt[0]), int(pt[1])), 1, color, -1)

            img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=img)
            img_label.config(image=photo)
            img_label._photo_ref = photo
            self._dirty = True

        tk.Label(controls, text="Canny Low (raise to suppress weak/shadow edges):").pack(anchor="w")
        low_scale = tk.Scale(controls, from_=0, to=255, orient="horizontal",
                              variable=low_var, command=redraw, length=320)
        low_scale.pack(fill="x")

        tk.Label(controls, text="Canny High (raise to keep only strong gradients):").pack(anchor="w")
        high_scale = tk.Scale(controls, from_=0, to=255, orient="horizontal",
                               variable=high_var, command=redraw, length=320)
        high_scale.pack(fill="x")

        contour_chk = tk.Checkbutton(controls, text="Show contour points (color = nesting depth: green->yellow->orange->red)",
                                      variable=show_contours_var, command=redraw)
        contour_chk.pack(anchor="w", pady=(2, 0))

        tk.Label(controls, text="Changes apply live to detection (canny_low/canny_high).",
                 fg="#666666", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        redraw()

    # ---------- Mouse drawing (only active in MODE_BOXING) ----------
    def on_mouse_down(self, event):
        if self.mode != self.MODE_BOXING:
            return
        self.drawing = True
        self.start_pt = (event.x, event.y)
        self.temp_box = (event.x, event.y, event.x, event.y)
        self._dirty = True

    def on_mouse_move(self, event):
        if self.mode != self.MODE_BOXING or not self.drawing or self.start_pt is None:
            return
        x0, y0 = self.start_pt
        self.temp_box = (x0, y0, event.x, event.y)
        self._dirty = True

    def on_mouse_up(self, event):
        if self.mode != self.MODE_BOXING or not self.drawing or self.start_pt is None:
            return

        x0, y0 = self.start_pt
        x1, y1 = event.x, event.y

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        x_min = max(0, min(self.W - 1, x_min))
        x_max = max(0, min(self.W - 1, x_max))
        y_min = max(0, min(self.H - 1, y_min))
        y_max = max(0, min(self.H - 1, y_max))

        if (x_max - x_min) > 8 and (y_max - y_min) > 8:
            self.boxes.append((x_min, y_min, x_max, y_max))
            self.box_counts.append(1)
            self._rebuild_box_count_ui()

        self.drawing = False
        self.start_pt = None
        self.temp_box = None
        self._dirty = True

    # ---------- Template building / detection (background thread) ----------
    def start_detection(self):
        if self._detect_busy:
            return
        if self.mode != self.MODE_BOXING or self.captured_frame is None:
            messagebox.showwarning("Warning", "Capture an image and draw boxes first.")
            return
        if not self.boxes:
            messagebox.showwarning("Warning", "Draw at least one bounding box.")
            return

        self._detect_busy = True
        self._set_buttons_enabled(False)
        self._last_debug_info = "Detecting... please wait."
        self._dirty = True

        frame_copy = self.captured_frame.copy()
        boxes_copy = list(self.boxes)
        counts_copy = [self.box_counts[i] if i < len(self.box_counts) else 1
                       for i in range(len(boxes_copy))]

        # Remember exactly what was drawn, so pressing "Back" from the
        # result screen can restore these same boxes/counts instead of
        # dropping the user into an empty boxing screen.
        self._last_used_boxes = list(boxes_copy)
        self._last_used_box_counts = list(counts_copy)

        t = threading.Thread(target=self._detect_worker, args=(frame_copy, boxes_copy, counts_copy),
                              daemon=True)
        t.start()
        self.root.after(50, self._poll_detect_queue)

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for b in (self.undo_btn, self.export_btn, self.debug_btn,
                  self.back_btn, self.start_btn, self.capture_btn,
                  self.pause_btn, self.rescan_btn, self.cam_select,
                  self.shape_thresh_scale, self.shape_thresh_entry):
            b.config(state=state)

    def _detect_worker(self, frame, boxes, counts):
        """Runs entirely off the Tk main thread so the UI stays responsive."""
        try:
            templates = []
            build_log = []
            for i, box in enumerate(boxes):
                tpl = self.build_template_from_box(frame, box, i)
                if tpl is not None:
                    tpl["expected_count"] = counts[i] if i < len(counts) else 1
                    templates.append(tpl)
                    build_log.append(
                        f"T{i}: area={tpl['area']:.0f}, expects {tpl['expected_count']} object(s)")
                else:
                    build_log.append(f"T{i}: FAILED (empty crop)")

            if not templates:
                self._detect_queue.put(("warn", "No valid template created."))
                return

            weak = [t for t in templates if t.get("contour") is None]
            weak_msg = None
            if weak:
                weak_msg = (f"{len(weak)} template(s) have no detectable contour/edge. "
                            "Try a tighter box, better lighting, or lower Canny thresholds.")

            t0 = time.time()
            self.templates = templates
            vis, dbg = self.detect_and_overlay(frame.copy())
            detect_ms = (time.time() - t0) * 1000.0
            debug_text = "\n".join(build_log) + "\n" + dbg

            self._detect_queue.put(("ok", (vis, debug_text, detect_ms, weak_msg)))
        except Exception:
            err = traceback.format_exc()
            self._detect_queue.put(("error", err))

    def _poll_detect_queue(self):
        try:
            kind, payload = self._detect_queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_detect_queue)
            return

        self._detect_busy = False
        self._set_buttons_enabled(True)

        if kind == "warn":
            messagebox.showwarning("Warning", payload)
            self._last_debug_info = ""
        elif kind == "error":
            self._last_debug_info = "ERROR:\n" + payload
            messagebox.showerror("Detection failed",
                                  "Loi khi detect, xem Debug Info panel de biet chi tiet.")
        else:  # "ok"
            vis, debug_text, detect_ms, weak_msg = payload
            self._last_detect_ms = detect_ms
            self._last_debug_info = debug_text
            self.result_frame = vis
            self.boxes = []
            self.box_counts = []
            self.temp_box = None
            self._rebuild_box_count_ui()
            self._set_mode(self.MODE_RESULT)
            if weak_msg:
                messagebox.showwarning("Weak template warning", weak_msg)
        self._dirty = True

    def _build_level_map(self, contours, hierarchy, root_idx):
        """BFS down the RETR_TREE hierarchy starting at root_idx, grouping
        every descendant contour by its nesting DEPTH relative to the root
        (root itself = level 0, its direct children = level 1, their
        children = level 2, and so on -- however many levels the image
        actually contains). Nodes below MIN_CONTOUR_AREA are dropped (and
        their own subtree is not traversed further, since a filtered-out
        contour's children are not meaningfully placeable either).

        Returns {level_int: [contour, ...]}.
        """
        level_map = {0: [contours[root_idx]]}
        frontier = [root_idx]
        level = 0
        while frontier:
            next_frontier = []
            for pidx in frontier:
                child = hierarchy[pidx][2]  # first_child
                while child != -1:
                    if cv2.contourArea(contours[child]) >= self.MIN_CONTOUR_AREA:
                        level_map.setdefault(level + 1, []).append(contours[child])
                        next_frontier.append(child)
                    child = hierarchy[child][0]  # next sibling
            frontier = next_frontier
            level += 1
        return level_map

    def build_template_from_box(self, frame, box, idx):
        """Build a shape template using RETR_TREE, which keeps EVERY
        nesting level of inner contours (holes, engravings, and anything
        nested even deeper) in addition to the outer silhouette -- unlike
        RETR_EXTERNAL, which discarded all of them.

        The template's "contour" (level 0 / outer) is still the single
        largest top-level contour in the ROI, exactly as before.
        Additionally, "level_map" holds a {level: [contours]} breakdown of
        every deeper nesting level actually present in the ROI, used by
        detect_and_overlay() to compute a depth-weighted shape distance.
        """
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or hierarchy is None:
            return {
                "id": idx, "box": box, "w": w, "h": h,
                "contour": None, "level_map": {}, "area": 0, "ref_len": 1.0,
            }

        hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

        # Top-level contours have parent == -1. Among those, the largest
        # (area-filtered) one is assumed to be the object's outer outline,
        # since the ROI is drawn tightly around ONE object.
        top_level_idx = [i for i, h4 in enumerate(hierarchy)
                          if h4[3] == -1 and cv2.contourArea(contours[i]) >= self.MIN_CONTOUR_AREA]

        if not top_level_idx:
            return {
                "id": idx, "box": box, "w": w, "h": h,
                "contour": None, "level_map": {}, "area": 0, "ref_len": 1.0,
            }

        parent_i = max(top_level_idx, key=lambda i: cv2.contourArea(contours[i]))
        best = contours[parent_i]
        area = cv2.contourArea(best)
        (rw, rh) = cv2.minAreaRect(best)[1]
        ref_len = max(rw, rh, 1.0)   # reference length used to derive scale later

        level_map = self._build_level_map(contours, hierarchy, parent_i)

        return {
            "id": idx,
            "box": box,
            "w": w,
            "h": h,
            "contour": best,
            "level_map": level_map,
            "area": area,
            "ref_len": ref_len,
        }

    # ---------- Detection ----------
    def _level_weight(self, n):
        """
        Weight for nesting level n:
            weight(n) = MAX_LEVEL_WEIGHT * e^(-n / sqrt(2))
        Level 0 (outer/parent contour) always gets the max weight
        (MAX_LEVEL_WEIGHT); deeper levels contribute progressively less,
        but decay MORE SLOWLY than the previous e^(-n) formula since
        n/sqrt(2) < n for every n > 0 -- so nested details now carry
        somewhat more relative influence than before.
        """
        return self.MAX_LEVEL_WEIGHT * math.exp(-n / math.sqrt(2))

    def _level_distance(self, tpl_contours_at_level, sc_contours_at_level):
        """Average best-match shape distance between all template contours
        at ONE nesting level and all scene contours at that SAME level.
        Each template contour is paired with its closest-matching scene
        contour (best-of), tolerating different ordering/count between
        template and scene at that level. Returns None if either side has
        no contours at this level or nothing was comparable."""
        if not tpl_contours_at_level or not sc_contours_at_level:
            return None
        per_best = []
        for tc in tpl_contours_at_level:
            best_d = None
            for sc in sc_contours_at_level:
                try:
                    d = cv2.matchShapes(tc, sc, cv2.CONTOURS_MATCH_I1, 0.0)
                except cv2.error:
                    continue
                if best_d is None or d < best_d:
                    best_d = d
            if best_d is not None:
                per_best.append(best_d)
        if not per_best:
            return None
        return sum(per_best) / len(per_best)

    def _weighted_shape_distance(self, tpl_level_map, sc_level_map):
        """Combines the shape distance at EVERY nesting level actually
        present into one weighted score:

            weight(n) = MAX_LEVEL_WEIGHT * e^(-n / sqrt(2))

        where n=0 is the outer/parent contour (max weight, always
        required) and each deeper level contributes progressively less.
        The number of levels used is whatever the template/scene tree
        actually has -- no hardcoded 2-level limit.

        Returns (weighted_distance, parent_distance), or None if the
        level-0 (outer) contours themselves cannot be compared at all.
        """
        tpl_l0 = tpl_level_map.get(0)
        sc_l0 = sc_level_map.get(0)
        if not tpl_l0 or not sc_l0:
            return None

        dist_parent = self._level_distance(tpl_l0, sc_l0)
        if dist_parent is None:
            return None

        weighted_sum = self._level_weight(0) * dist_parent
        weight_total = self._level_weight(0)

        # Walk every deeper level that the TEMPLATE actually has contours
        # at (however many levels that turns out to be); levels missing on
        # the scene side simply do not contribute (no penalty, no bonus).
        deeper_levels = sorted(n for n in tpl_level_map.keys() if n > 0)
        for n in deeper_levels:
            d_n = self._level_distance(tpl_level_map.get(n), sc_level_map.get(n))
            if d_n is None:
                continue
            w_n = self._level_weight(n)
            weighted_sum += w_n * d_n
            weight_total += w_n

        weighted = weighted_sum / weight_total
        return weighted, dist_parent

    def detect_and_overlay(self, frame):
        """Contour / shape matching over the WHOLE scene, using RETR_TREE
        so EVERY nesting level of inner contours (holes, engravings, and
        anything nested even deeper) is available, matching however many
        levels the image actually contains -- no hardcoded 2-level limit.

        Pipeline per template:
          1. Canny + findContours(RETR_TREE) on the FULL scene (once) ->
             build a per-candidate level map: level 0 = the top-level
             contour (candidate object outline), level 1 = its direct
             children, level 2 = their children, etc., for as many levels
             as actually exist under that candidate.
          2. For each scene candidate: compute a depth-weighted shape
             distance against the template, where each level n's
             contribution is weighted by weight(n) = MAX_LEVEL_WEIGHT *
             e^(-n / sqrt(2)) -- level 0 (outline) always weighs the most,
             deeper levels contribute progressively less. 0 = identical.
          3. Convert to quality = 1 / (1 + weighted_distance) for
             consistent UI reporting (1.0 = perfect match, ->0 = very
             different).
          4. Reject any candidate whose weighted distance > MAX_SHAPE_DISTANCE.
          5. Sort remaining candidates by quality (best first), take up to
             `expected_count` of them per template.
          6. Derive position (centroid), scale (vs template's reference
             length) and angle (minAreaRect orientation) directly from each
             accepted scene contour -- no homography needed.
        """
        vis = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_full = cv2.GaussianBlur(gray_full, (5, 5), 0)
        edges_full = cv2.Canny(blur_full, self.canny_low, self.canny_high)
        edges_full = cv2.dilate(edges_full, np.ones((3, 3), np.uint8), iterations=1)

        raw_contours, hierarchy = cv2.findContours(edges_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        all_results = []
        dbg_lines = []

        if not raw_contours or hierarchy is None:
            dbg_lines.append("Scene: no contours found.")
            self.machine_paths = []
            self.apply_glue_program([])
            return vis, "\n".join(dbg_lines)

        hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

        top_level_idx = [i for i, h4 in enumerate(hierarchy)
                          if h4[3] == -1 and cv2.contourArea(raw_contours[i]) >= self.MIN_CONTOUR_AREA]

        # scene_candidates: list of (top_level_contour, level_map)
        scene_candidates = []
        for pi in top_level_idx:
            level_map = self._build_level_map(raw_contours, hierarchy, pi)
            scene_candidates.append((raw_contours[pi], level_map))

        if not scene_candidates:
            dbg_lines.append("Scene: no contours found.")
            self.machine_paths = []
            self.apply_glue_program([])
            return vis, "\n".join(dbg_lines)

        dbg_lines.append(f"Scene: {len(scene_candidates)} candidate contour(s) found.")

        for tpl in self.templates:
            expected = max(1, int(tpl.get("expected_count", 1)))

            if tpl.get("contour") is None:
                dbg_lines.append(f"Template {tpl['id']}: no template contour, skipped.")
                continue

            tpl_level_map = tpl.get("level_map", {0: [tpl["contour"]]})
            num_levels = len(tpl_level_map)

            candidates = []
            for sc, sc_level_map in scene_candidates:
                res = self._weighted_shape_distance(tpl_level_map, sc_level_map)
                if res is None:
                    continue
                dist, parent_dist = res
                quality = 1.0 / (1.0 + dist)
                candidates.append((dist, quality, sc, parent_dist))

            if not candidates:
                dbg_lines.append(f"Template {tpl['id']}: no comparable candidates, skipped.")
                continue

            candidates.sort(key=lambda t: t[0])  # smallest weighted distance first

            dbg_lines.append(
                f"Template {tpl['id']}: {num_levels} nesting level(s) in template, "
                f"best candidate distance={candidates[0][0]:.3f} "
                f"(outline-only={candidates[0][3]:.3f}, quality={candidates[0][1]:.2f}), "
                f"wants up to {expected} object(s)"
            )

            accepted = 0
            for dist, quality, sc, parent_dist in candidates:
                if accepted >= expected:
                    break
                if dist > self.MAX_SHAPE_DISTANCE:
                    dbg_lines.append(
                        f"Template {tpl['id']}: next best candidate distance={dist:.3f} > "
                        f"ceiling ({self.MAX_SHAPE_DISTANCE:.2f}), stopping (found {accepted}/{expected})"
                    )
                    break

                (cx, cy), (rw, rh), rangle = cv2.minAreaRect(sc)
                # Normalize angle to the LONGER side, for a stable "object
                # orientation" reading regardless of OpenCV's rect convention.
                if rw < rh:
                    rangle += 90.0
                    rw, rh = rh, rw
                scale = max(rw, rh) / tpl["ref_len"] if tpl.get("ref_len") else 1.0

                box_pts = cv2.boxPoints(((cx, cy), (rw, rh), rangle)).astype(np.int32)
                cv2.polylines(vis, [box_pts], True, (0, 0, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                cv2.polylines(vis, [sc.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

                all_results.append({
                    "template_id": tpl["id"],
                    "position": [float(cx), float(cy)],
                    "scale": float(scale),
                    "angle_deg": float(rangle),
                    "shape_distance": float(dist),
                    "outline_shape_distance": float(parent_dist),
                    "quality": float(quality),
                    "edge_points": sc.reshape(-1, 2).astype(np.float32).tolist(),
                })
                accepted += 1

            dbg_lines.append(f"Template {tpl['id']}: accepted {accepted} instance(s)")

        all_results.sort(key=lambda r: r["quality"], reverse=True)

        self.machine_paths = all_results
        self.apply_glue_program(all_results)
        return vis, "\n".join(dbg_lines)

    def apply_glue_program(self, machine_paths):
        pass

    # ---------- Drawing ----------
    def draw_boxes_on_image(self, img):
        out = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, f"T{i}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.temp_box is not None:
            x1, y1, x2, y2 = self.temp_box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
        return out

    def show_frame_on_canvas(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if self.photo is None:
            self.photo = ImageTk.PhotoImage(image=img)
            self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        else:
            new_photo = ImageTk.PhotoImage(image=img)
            self.canvas.itemconfig(self._canvas_img_id, image=new_photo)
            self.photo = new_photo

    def update_info_panel(self):
        mode_text = {
            self.MODE_LIVE: "Live",
            self.MODE_BOXING: "Frozen (drawing boxes)",
            self.MODE_RESULT: "Static result (captured image)",
        }[self.mode]

        self.status_vars["Mode"].config(text=mode_text)
        self.status_vars["Detect time"].config(text=f"{self._last_detect_ms:.1f} ms")
        self.status_vars["Boxes drawn"].config(text=str(len(self.boxes)))
        self.status_vars["Templates"].config(text=str(len(self.templates)))
        self.status_vars["Detections"].config(text=str(len(self.machine_paths)))
        self.status_vars["Max shape dist"].config(text=f"{self.MAX_SHAPE_DISTANCE:.2f}")
        self.debug_lbl.config(text=self._last_debug_info)

        for row in self.detect_table.get_children():
            self.detect_table.delete(row)
        for res in self.machine_paths:
            self.detect_table.insert("", "end", values=(
                res["template_id"],
                f"{res['position'][0]:.0f}",
                f"{res['position'][1]:.0f}",
                f"{res['angle_deg']:.1f}",
                f"{res['scale']:.2f}",
                f"{res.get('quality', 0):.2f}",
            ))

    # ---------- Main loop ----------
    def update_loop(self):
        # CRITICAL FIX: camera is read ONLY in MODE_LIVE. In MODE_BOXING and
        # MODE_RESULT the camera is never touched, so nothing can overwrite
        # the captured still image or interfere with the static result.
        # PAUSE: when self.paused is True, camera .read() is skipped entirely
        # (even in MODE_LIVE), freezing the preview on the last grabbed frame.
        if self.mode == self.MODE_LIVE and not self.paused:
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                ret, frame = False, None
            if ret:
                frame = cv2.resize(frame, (self.W, self.H))
                self.last_live_frame = frame
                self._dirty = True  # live video always needs a repaint
            display = self.last_live_frame if self.last_live_frame is not None else \
                np.zeros((self.H, self.W, 3), dtype=np.uint8)

        elif self.mode == self.MODE_LIVE and self.paused:
            display = self.last_live_frame if self.last_live_frame is not None else \
                np.zeros((self.H, self.W, 3), dtype=np.uint8)

        elif self.mode == self.MODE_BOXING:
            display = self.draw_boxes_on_image(self.captured_frame)

        else:  # MODE_RESULT
            display = self.result_frame if self.result_frame is not None else self.captured_frame

        # Optimization: only touch the canvas / info panel when something
        # actually changed, instead of rebuilding a PhotoImage every 15ms
        # even while the UI is completely static (MODE_BOXING/MODE_RESULT).
        if self._dirty:
            self.show_frame_on_canvas(display)
            self.update_info_panel()
            self._dirty = False

        self._after_id = self.root.after(15, self.update_loop)

    def on_close(self):
        """Used both by the window's close button (WM_DELETE_WINDOW) and by
        the 'Exit' button. Confirms with the user, cancels any pending
        after() callback first, then releases the camera before destroying
        the window -- avoids a callback firing on an already-released
        VideoCapture."""
        if self._detect_busy:
            if not messagebox.askyesno("Confirm exit",
                                        "A detection job is still running. Exit anyway?"):
                return
        else:
            if not messagebox.askyesno("Confirm exit", "Exit the program?"):
                return

        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        if self._shape_thresh_after_id is not None:
            try:
                self.root.after_cancel(self._shape_thresh_after_id)
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = GlueVisionApp(root)
    except SystemExit:
        raise
    root.mainloop()