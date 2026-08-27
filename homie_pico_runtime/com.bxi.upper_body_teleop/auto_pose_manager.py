#!/usr/bin/env python3
"""Run the SONIC PICO pose pipeline with automatic post-calibration entry.

This Mod only needs calibrated POSE packets. It deliberately omits SONIC's
planner modes: ABXY requests calibration, successful calibration enters POSE,
and grip values in the POSE stream gate the two arms downstream.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import socket
import sys
import sysconfig
import threading
import time
from types import ModuleType


CONFIG_ERROR = getattr(os, "EX_CONFIG", 78)
XRT_SERVICE_HOST = "127.0.0.1"
XRT_SERVICE_PORT = 60061
XRT_SERVICE_PROBE_TIMEOUT_S = 1.0
XRT_INIT_WATCHDOG_S = 5.0
PINOCCHIO_API = (
    "SE3",
    "buildModelFromUrdf",
    "forwardKinematics",
    "updateFramePlacements",
    "neutral",
    "integrate",
)


class XrtServiceConflictError(RuntimeError):
    """A single-instance marker exists without a reachable service."""


class PicoPortInUseError(RuntimeError):
    """The configured PICO ZMQ publisher port already has a listener."""


@dataclass(slots=True)
class AutoPoseController:
    """Small, hardware-independent OFF/calibrating/POSE state machine."""

    pose_mode: int = 1
    off_mode: int = 0
    mode: int = 0
    calibration_requested: bool = False

    def update(self, *, abxy_rising: bool, calibration_succeeded: bool) -> bool:
        """Update the mode and return whether it changed.

        Face-button subsets such as A+X are intentionally absent from this
        interface. Per-arm activation belongs to the streamed grip values.
        """
        previous = self.mode
        if abxy_rising:
            self.mode = self.off_mode
            self.calibration_requested = True
        if self.calibration_requested and calibration_succeeded:
            self.calibration_requested = False
            self.mode = self.pose_mode
        return self.mode != previous


class _ExternalXrtSession:
    """Own only an SDK client connected to an already-running PC Service."""

    def __init__(self, manager: ModuleType, prefix: str) -> None:
        self._manager = manager
        self._prefix = prefix
        self._sdk_initialized = False
        self._closed = False

    def start(self, stop_event: threading.Event) -> bool:
        init_done = threading.Event()

        def _watch_init() -> None:
            while not init_done.wait(timeout=XRT_INIT_WATCHDOG_S):
                print(
                    f"[{self._prefix}] xrt.init() is still running while "
                    "connecting to the existing RoboticsServiceProcess",
                    flush=True,
                )

        threading.Thread(target=_watch_init, daemon=True).start()
        started = time.monotonic()
        try:
            self._manager.xrt.init()
            self._sdk_initialized = True
        finally:
            init_done.set()
        print(
            f"[{self._prefix}] connected to existing RoboticsServiceProcess "
            f"in {time.monotonic() - started:.2f}s",
            flush=True,
        )
        return not stop_event.is_set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sdk_initialized:
            try:
                self._manager.xrt.close()
            finally:
                self._sdk_initialized = False


def _xrt_service_is_reachable(
    host: str = XRT_SERVICE_HOST,
    port: int = XRT_SERVICE_PORT,
) -> bool:
    try:
        with socket.create_connection(
            (host, port), timeout=XRT_SERVICE_PROBE_TIMEOUT_S
        ):
            return True
    except OSError:
        return False


def _publisher_port_is_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _start_xrt_session(manager: ModuleType, stop_event: threading.Event):
    """Start an owned service, or safely reuse a confirmed external instance."""
    try:
        return manager._start_xrt_session("UpperBodyPose", stop_event=stop_event)
    except RuntimeError as exc:
        duplicate_marker = "exited during startup with code 0"
        if duplicate_marker not in str(exc):
            raise
        if not _xrt_service_is_reachable():
            raise XrtServiceConflictError(
                "RoboticsServiceProcess reported another instance, but "
                f"{XRT_SERVICE_HOST}:{XRT_SERVICE_PORT} is not reachable; "
                "stop the stale PC Service instance before retrying"
            ) from exc

        print(
            "[UpperBodyPose] existing RoboticsServiceProcess detected on "
            f"{XRT_SERVICE_HOST}:{XRT_SERVICE_PORT}; reusing it without ownership",
            flush=True,
        )
        session = _ExternalXrtSession(manager, "UpperBodyPose")
        try:
            if not session.start(stop_event):
                session.close()
                return None
        except BaseException:
            session.close()
            raise
        atexit.register(session.close)
        return session


def _sonic_root() -> Path:
    configured = os.environ.get("BXI_MOD_ROOT", "").strip()
    root = (
        Path(configured).expanduser().absolute()
        if configured
        else Path(__file__).resolve().parent.parent / "com.bxi.sonic"
    )
    manifest = root / "mod.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"SONIC Mod is unavailable: {manifest}")
    return root


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_robotics_pinocchio(pin: ModuleType) -> None:
    missing = tuple(
        name for name in PINOCCHIO_API if not callable(getattr(pin, name, None))
    )
    if missing:
        raise RuntimeError(
            "incompatible Python package 'pinocchio' at "
            f"{getattr(pin, '__file__', '<unknown>')}; missing robotics API: "
            + ", ".join(missing)
        )


def _cmeel_pinocchio_paths() -> tuple[Path, ...]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if not value:
            continue
        candidate = Path(value) / "cmeel.prefix" / "lib" / version / "site-packages"
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def _activate_robotics_pinocchio() -> ModuleType:
    """Prefer cmeel's robotics bindings over the unrelated PyPI namesake."""
    cmeel_paths = _cmeel_pinocchio_paths()
    for path in reversed(cmeel_paths):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    loaded = sys.modules.get("pinocchio")
    if loaded is not None:
        try:
            _validate_robotics_pinocchio(loaded)
            return loaded
        except RuntimeError:
            for name in tuple(sys.modules):
                if name == "pinocchio" or name.startswith("pinocchio."):
                    sys.modules.pop(name, None)

    importlib.invalidate_caches()
    try:
        pin = importlib.import_module("pinocchio")
    except ImportError as exc:
        searched = ", ".join(str(path) for path in cmeel_paths) or "none found"
        raise RuntimeError(
            "robotics Pinocchio cannot be imported; cmeel search paths: " + searched
        ) from exc
    _validate_robotics_pinocchio(pin)
    print(
        "[UpperBodyPose] selected robotics Pinocchio "
        f"{getattr(pin, '__version__', 'unknown')} "
        f"({getattr(pin, '__file__', '<unknown>')})",
        flush=True,
    )
    return pin


def _load_sonic_runtime() -> ModuleType:
    sonic_root = _sonic_root()
    pico_root = sonic_root / "pico"
    launcher_path = pico_root / "manager_launcher.py"
    if not launcher_path.is_file():
        raise FileNotFoundError(f"SONIC PICO launcher is unavailable: {launcher_path}")
    root_text = str(pico_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    launcher = _load_module("_upper_body_teleop_sonic_launcher", launcher_path)
    launcher.prepare_service_environment()
    launcher.reexec_if_needed("upper_body_auto_pose", launcher.MANAGER_IMPORTS)
    launcher._validate_manager_runtime()
    _activate_robotics_pinocchio()

    vendor_root = launcher._vendor_root()
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from gear_sonic.scripts import pico_manager_thread_server as manager

    required = (
        "FaceComboEdgeDetector",
        "MANAGER_POLL_PERIOD_SECONDS",
        "PICO_BODY_SAMPLE_MAX_AGE_SECONDS",
        "PICO_BODY_WAIT_LOG_SECONDS",
        "PicoReader",
        "PoseStreamer",
        "StreamMode",
        "ThreePointPose",
        "_install_stop_signal_handlers",
        "_restore_signal_handlers",
        "_start_xrt_session",
        "build_command_message",
        "get_abxy_buttons",
        "get_controller_inputs",
        "np",
        "pack_pose_message",
        "zmq",
    )
    missing = tuple(name for name in required if not hasattr(manager, name))
    if missing:
        raise RuntimeError(
            "SONIC PICO runtime is incompatible; missing API: " + ", ".join(missing)
        )
    if manager.StreamMode.OFF.value != 0 or manager.StreamMode.POSE.value != 1:
        raise RuntimeError("SONIC PICO stream-mode values are incompatible")
    return manager


def _send_command(manager: ModuleType, socket, *, running: bool) -> None:
    socket.send(
        manager.build_command_message(
            start=running,
            stop=not running,
            planner=False,
        )
    )


def _create_calibration_provider():
    from gear_sonic.utils.teleop.calibration import create_calibration_provider

    return create_calibration_provider()


def run_auto_pose_manager(manager: ModuleType, args: argparse.Namespace) -> None:
    """Reuse SONIC acquisition/calibration/POSE classes with a smaller UI."""
    stop_event = args.stop_event
    if not _publisher_port_is_available(args.port):
        raise PicoPortInUseError(
            f"PICO ZMQ port 0.0.0.0:{args.port} is already in use; "
            "stop the other SONIC/PICO manager before entering this state"
        )
    # Load and validate native calibration before acquiring XRT/ZMQ resources.
    calibration_provider = _create_calibration_provider()
    print(
        f"[UpperBodyPose] PICO calibration profile: {calibration_provider.name}",
        flush=True,
    )
    xrt_session = None
    context = None
    publisher = None
    reader = None
    three_point = None
    pose_streamer = None
    controller = None
    try:
        xrt_session = _start_xrt_session(manager, stop_event)
        if xrt_session is None:
            print("[UpperBodyPose] shutdown completed during XRT startup", flush=True)
            return

        context = manager.zmq.Context()
        publisher = context.socket(manager.zmq.PUB)
        publisher.setsockopt(manager.zmq.LINGER, 0)
        publisher.bind(f"tcp://*:{args.port}")
        time.sleep(0.1)
        print(f"[UpperBodyPose] ZMQ socket bound to port {args.port}", flush=True)

        reader = manager.PicoReader(max_queue_size=args.buffer_size)
        reader.start()
        three_point = manager.ThreePointPose(
            enable_vis_vr3pt=False,
            with_g1_robot=False,
            enable_waist_tracking=False,
            enable_smpl_vis=False,
            log_prefix="UpperBodyPose",
            calibration_provider=calibration_provider,
        )
        pose_streamer = manager.PoseStreamer(
            socket=publisher,
            reader=reader,
            three_point=three_point,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            record_dir=args.record_dir,
            record_format=args.record_format,
            log_prefix="UpperBodyPose",
        )
        controller = AutoPoseController(
            pose_mode=manager.StreamMode.POSE.value,
            off_mode=manager.StreamMode.OFF.value,
        )
        combo_edges = manager.FaceComboEdgeDetector()
        previous_buttons = None
        last_calibration_wait_log = None
        prev_toggle_dc = False
        prev_toggle_da = False

        print(
            "[UpperBodyPose] controls: ABXY=calibrate and enter POSE; "
            "left/right grip gate each arm; A+X is unused",
            flush=True,
        )
        _send_command(manager, publisher, running=False)
        while not stop_event.is_set():
            loop_started = time.monotonic()
            a_pressed, b_pressed, x_pressed, y_pressed = manager.get_abxy_buttons()
            buttons = (a_pressed, b_pressed, x_pressed, y_pressed)
            if buttons != previous_buttons:
                pressed_names = "+".join(
                    name
                    for name, pressed in zip(("A", "B", "X", "Y"), buttons)
                    if pressed
                )
                print(
                    f"[UpperBodyPose] face buttons: {pressed_names or 'released'}",
                    flush=True,
                )
                previous_buttons = buttons

            _, _, _, left_grip, _ = manager.get_controller_inputs()
            combo = combo_edges.update(*buttons)
            previous_mode = controller.mode
            controller.update(
                abxy_rising=combo.start_rising,
                calibration_succeeded=False,
            )
            if combo.rearmed:
                print("[UpperBodyPose] ABXY re-armed after full release", flush=True)
            if combo.start_rising:
                if previous_mode == manager.StreamMode.POSE.value:
                    pose_streamer.on_mode_exit()
                _send_command(manager, publisher, running=False)
                last_calibration_wait_log = None
                print("[UpperBodyPose] ABXY accepted; calibration requested", flush=True)

            if controller.calibration_requested:
                sample = reader.get_latest()
                sample_age = (
                    time.monotonic() - float(sample["timestamp_monotonic"])
                    if sample is not None
                    else float("inf")
                )
                fresh_sample = (
                    sample is not None
                    and sample_age <= manager.PICO_BODY_SAMPLE_MAX_AGE_SECONDS
                )
                calibration_succeeded = bool(
                    fresh_sample
                    and three_point.calibrate_now(sample["body_poses_np"])
                )
                changed = controller.update(
                    abxy_rising=False,
                    calibration_succeeded=calibration_succeeded,
                )
                if changed:
                    pose_streamer.reset_yaw()
                    _send_command(manager, publisher, running=True)
                    print(
                        "[UpperBodyPose] calibration complete; entering POSE automatically",
                        flush=True,
                    )
                elif not fresh_sample:
                    now = time.monotonic()
                    if (
                        last_calibration_wait_log is None
                        or now - last_calibration_wait_log
                        >= manager.PICO_BODY_WAIT_LOG_SECONDS
                    ):
                        age_text = (
                            "not received"
                            if sample is None
                            else f"stale by {sample_age:.2f}s"
                        )
                        print(
                            "[UpperBodyPose] calibration queued but PICO has no fresh "
                            f"body frame ({age_text}); enable Body Tracking and press "
                            "Send in the headset app",
                            flush=True,
                        )
                        last_calibration_wait_log = now
                elif not calibration_succeeded:
                    now = time.monotonic()
                    if (
                        last_calibration_wait_log is None
                        or now - last_calibration_wait_log
                        >= manager.PICO_BODY_WAIT_LOG_SECONDS
                    ):
                        print(
                            "[UpperBodyPose] calibration frame was rejected; waiting "
                            "for the next valid body frame",
                            flush=True,
                        )
                        last_calibration_wait_log = now

            if controller.mode == manager.StreamMode.POSE.value:
                pose_streamer.run_once()

            toggle_dc_raw = bool(a_pressed) and left_grip > 0.5
            toggle_da_raw = bool(b_pressed) and left_grip > 0.5
            toggle_dc = toggle_dc_raw and not prev_toggle_dc
            toggle_da = toggle_da_raw and not prev_toggle_da
            prev_toggle_dc = toggle_dc_raw
            prev_toggle_da = toggle_da_raw
            publisher.send(
                manager.pack_pose_message(
                    {
                        "stream_mode": manager.np.array(
                            [controller.mode], dtype=manager.np.int32
                        ),
                        "calibration_ready": manager.np.array(
                            [three_point.is_calibrated], dtype=bool
                        ),
                        "toggle_data_collection": manager.np.array(
                            [toggle_dc], dtype=bool
                        ),
                        "toggle_data_abort": manager.np.array(
                            [toggle_da], dtype=bool
                        ),
                    },
                    topic="manager_state",
                )
            )

            remaining = manager.MANAGER_POLL_PERIOD_SECONDS - (
                time.monotonic() - loop_started
            )
            if remaining > 0.0:
                stop_event.wait(remaining)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if (
            controller is not None
            and pose_streamer is not None
            and controller.mode == manager.StreamMode.POSE.value
        ):
            try:
                pose_streamer.on_mode_exit()
            except Exception as exc:
                print(
                    f"[UpperBodyPose] warning: failed to exit POSE mode: {exc}",
                    flush=True,
                )
        cleanup_steps = []
        if reader is not None:
            cleanup_steps.append(("PICO reader", reader.stop))
        if three_point is not None:
            cleanup_steps.append(("three-point processor", three_point.close))
        if publisher is not None:
            cleanup_steps.append(
                ("ZMQ publisher", lambda: publisher.close(linger=0))
            )
        if context is not None:
            cleanup_steps.append(("ZMQ context", context.term))
        if xrt_session is not None:
            cleanup_steps.append(("XRT session", xrt_session.close))
        for resource_name, close_resource in cleanup_steps:
            try:
                close_resource()
            except Exception as exc:
                print(
                    f"[UpperBodyPose] warning: failed to close {resource_name}: {exc}",
                    flush=True,
                )
        print("[UpperBodyPose] shutdown complete", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--num_frames_to_send", type=int, default=5)
    parser.add_argument("--target_fps", type=int, default=50)
    parser.add_argument("--record_dir", type=str, default="")
    parser.add_argument("--record_format", type=str, default="npz")
    return parser


def main() -> int:
    try:
        manager = _load_sonic_runtime()
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"[UpperBodyPose] configuration error: {exc}", file=sys.stderr, flush=True)
        return CONFIG_ERROR

    args = _parser().parse_args()
    args.stop_event = threading.Event()
    previous_handlers = manager._install_stop_signal_handlers(
        args.stop_event, "UpperBodyPose"
    )
    try:
        try:
            run_auto_pose_manager(manager, args)
        except (PicoPortInUseError, XrtServiceConflictError) as exc:
            print(f"[UpperBodyPose] configuration error: {exc}", file=sys.stderr)
            return CONFIG_ERROR
    finally:
        manager._restore_signal_handlers(previous_handlers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
