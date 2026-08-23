import math
import queue
import threading


COMMAND_LIMITS = {
    "x_vel": (-0.8, 1.2),
    "y_vel": (-0.5, 0.5),
    "yaw_vel": (-0.8, 0.8),
    "height": (0.30, 1.01),
}


class SliderCommand:
    def __init__(
        self,
        *,
        x_vel=0.0,
        y_vel=0.0,
        yaw_vel=0.0,
        height=0.8,
        limits=None,
        mode_limits=None,
        mode=None,
    ):
        source_limits = COMMAND_LIMITS if limits is None else limits
        self.limits = self._normalize_limits(source_limits)
        self.mode_limits = {}
        if mode_limits:
            self.mode_limits = {
                str(mode_name): self._normalize_limits(mode_values)
                for mode_name, mode_values in mode_limits.items()
            }
            if mode is None:
                mode = next(iter(self.mode_limits))
            if mode not in self.mode_limits:
                raise ValueError(f"Unsupported initial command mode: {mode}")
        elif mode is not None:
            raise ValueError("A command mode requires mode-specific limits")
        self._mode = mode

        initial = {
            "x_vel": float(x_vel),
            "y_vel": float(y_vel),
            "yaw_vel": float(yaw_vel),
            "height": float(height),
        }
        for name, value in initial.items():
            lower, upper = self._active_limits()[name]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"Initial {name}={value} is outside [{lower}, {upper}]"
                )

        self._initial = initial
        self._initial_mode = mode
        self._values = dict(initial)
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_limits(source_limits):
        if set(source_limits) != set(COMMAND_LIMITS):
            raise ValueError(
                f"Command limits must define exactly {tuple(COMMAND_LIMITS)}"
            )
        limits = {}
        for name in COMMAND_LIMITS:
            lower, upper = (float(value) for value in source_limits[name])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"Invalid limits for {name}: [{lower}, {upper}]")
            limits[name] = (lower, upper)
        return limits

    def _active_limits(self):
        if self._mode is None:
            return self.limits
        return self.mode_limits[self._mode]

    def _clamp(self, name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        lower, upper = self._active_limits()[name]
        return min(upper, max(lower, value))

    def set_values(self, *, x_vel=None, y_vel=None, yaw_vel=None, height=None):
        updates = {
            "x_vel": x_vel,
            "y_vel": y_vel,
            "yaw_vel": yaw_vel,
            "height": height,
        }
        with self._lock:
            for name, value in updates.items():
                if value is not None:
                    self._values[name] = self._clamp(name, value)

    def reset_speeds(self):
        self.set_values(x_vel=0.0, y_vel=0.0, yaw_vel=0.0)

    @property
    def mode_names(self):
        return tuple(self.mode_limits)

    def mode_snapshot(self):
        with self._lock:
            return self._mode

    def active_limits(self):
        with self._lock:
            return dict(self._active_limits())

    def set_mode(self, mode):
        with self._lock:
            if mode not in self.mode_limits:
                raise ValueError(f"Unsupported command mode: {mode}")
            self._mode = mode
            for name, value in self._values.items():
                self._values[name] = self._clamp(name, value)

    def restore_defaults(self):
        with self._lock:
            self._mode = self._initial_mode
            self._values = dict(self._initial)

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def state_snapshot(self):
        with self._lock:
            return dict(self._values), self._mode


def apply_command_snapshot(env, command):
    if env.commands.shape[1] <= 4:
        raise RuntimeError("Interactive ELF3 commands require five command columns")
    env.commands[:, 0] = command["x_vel"]
    env.commands[:, 1] = command["y_vel"]
    env.commands[:, 2] = command["yaw_vel"]
    env.commands[:, 4] = command["height"]
    if hasattr(env, "height_command_targets"):
        env.height_command_targets[:] = command["height"]


class SliderControlPanel:
    def __init__(self, command):
        self.command = command
        self._errors = queue.Queue()
        self._ready = threading.Event()
        self._thread = None

    def start(self, stop_event, timeout=5.0):
        if self._thread is not None:
            raise RuntimeError("Slider control panel has already been started")
        self._thread = threading.Thread(
            target=self._run,
            args=(stop_event,),
            name="elf3-command-panel",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout):
            stop_event.set()
            raise RuntimeError("Timed out while starting the Tkinter control panel")
        self.raise_if_failed()

    def raise_if_failed(self):
        try:
            details = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError(
            "Tkinter control panel failed; verify tkinter and the graphical "
            f"display ({details})"
        )

    def join(self, timeout=2.0):
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, stop_event):
        import gc

        self._run_tk(stop_event)
        gc.collect()

    def _run_tk(self, stop_event):
        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title("ELF3 command control")
            root.geometry("700x395+40+80" if self.command.mode_names else "700x345+40+80")
            root.resizable(False, False)
            root.attributes("-topmost", True)
            root.columnconfigure(0, weight=1)
            root.rowconfigure(0, weight=1)

            style = ttk.Style(root)
            style.configure("Panel.TFrame", padding=18)
            style.configure("Title.TLabel", font=("TkDefaultFont", 14, "bold"))

            panel = ttk.Frame(root, style="Panel.TFrame")
            panel.grid(row=0, column=0, sticky="nsew")
            panel.columnconfigure(1, weight=1)
            ttk.Label(
                panel, text="ELF3 command control", style="Title.TLabel"
            ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

            variables = {}
            value_labels = {}
            sliders = {}
            units = {
                "x_vel": "m/s",
                "y_vel": "m/s",
                "yaw_vel": "rad/s",
                "height": "m",
            }

            def set_command_value(name, raw_value):
                self.command.set_values(**{name: float(raw_value)})
                value = self.command.snapshot()[name]
                value_labels[name].configure(text=f"{value:+.2f} {units[name]}")

            def add_slider(row, name, label):
                lower, upper = self.command.active_limits()[name]
                value = self.command.snapshot()[name]
                ttk.Label(panel, text=label).grid(
                    row=row, column=0, sticky="w", padx=(0, 16), pady=8
                )
                variable = tk.DoubleVar(value=value)
                slider = ttk.Scale(
                    panel,
                    from_=lower,
                    to=upper,
                    orient="horizontal",
                    length=360,
                    variable=variable,
                    command=lambda raw, key=name: set_command_value(key, raw),
                )
                slider.grid(row=row, column=1, sticky="ew", pady=8)
                if lower == upper:
                    slider.state(["disabled"])
                value_label = ttk.Label(panel, width=13, anchor="e")
                value_label.grid(
                    row=row, column=2, sticky="e", padx=(16, 0), pady=8
                )
                variables[name] = variable
                value_labels[name] = value_label
                sliders[name] = slider
                set_command_value(name, value)

            slider_start_row = 1
            mode_variable = None
            if self.command.mode_names:
                mode_variable = tk.StringVar(value=self.command.mode_snapshot())
                ttk.Label(panel, text="Control mode").grid(
                    row=1, column=0, sticky="w", padx=(0, 16), pady=8
                )
                mode_buttons = ttk.Frame(panel)
                mode_buttons.grid(row=1, column=1, columnspan=2, sticky="w", pady=8)
                slider_start_row = 2

            add_slider(slider_start_row, "x_vel", "Backward / forward (vx)")
            add_slider(slider_start_row + 1, "y_vel", "Right / left (vy)")
            add_slider(slider_start_row + 2, "yaw_vel", "Right / left turn (yaw)")
            add_slider(slider_start_row + 3, "height", "Body height")

            def refresh_widgets(names):
                values = self.command.snapshot()
                for name in names:
                    variables[name].set(values[name])
                    value_labels[name].configure(
                        text=f"{values[name]:+.2f} {units[name]}"
                    )

            def refresh_limits():
                limits = self.command.active_limits()
                for name, slider in sliders.items():
                    lower, upper = limits[name]
                    slider.configure(from_=lower, to=upper)
                    slider.state(["disabled"] if lower == upper else ["!disabled"])
                refresh_widgets(limits)

            def change_mode():
                self.command.set_mode(mode_variable.get())
                refresh_limits()

            if self.command.mode_names:
                for column, mode_name in enumerate(self.command.mode_names):
                    ttk.Radiobutton(
                        mode_buttons,
                        text=mode_name.capitalize(),
                        value=mode_name,
                        variable=mode_variable,
                        command=change_mode,
                    ).grid(row=0, column=column, padx=(0, 12))

            def reset_speeds():
                self.command.reset_speeds()
                refresh_widgets(("x_vel", "y_vel", "yaw_vel"))

            def restore_defaults():
                self.command.restore_defaults()
                if mode_variable is not None:
                    mode_variable.set(self.command.mode_snapshot())
                refresh_limits()

            ttk.Separator(panel, orient="horizontal").grid(
                row=slider_start_row + 4,
                column=0,
                columnspan=3,
                sticky="ew",
                pady=(10, 12),
            )
            buttons = ttk.Frame(panel)
            buttons.grid(
                row=slider_start_row + 5, column=0, columnspan=3, sticky="e"
            )
            ttk.Button(buttons, text="Zero speeds", command=reset_speeds).grid(
                row=0, column=0, padx=(0, 8)
            )
            ttk.Button(
                buttons, text="Restore defaults", command=restore_defaults
            ).grid(row=0, column=1)

            def close():
                stop_event.set()
                if root.winfo_exists():
                    root.destroy()

            def check_simulation():
                if stop_event.is_set():
                    close()
                elif root.winfo_exists():
                    root.after(100, check_simulation)

            root.protocol("WM_DELETE_WINDOW", close)
            root.after(100, check_simulation)
            self._ready.set()
            root.mainloop()
        except BaseException as error:
            self._errors.put(f"{type(error).__name__}: {error}")
            stop_event.set()
        finally:
            self._ready.set()
