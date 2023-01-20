import multiprocessing
import multiprocessing.synchronize
import os
import pathlib
import queue
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Literal

import TMSiFileFormats
import TMSiSDK
from pynput.keyboard import Key, Listener

import realtime_decoding

from .helpers import _PathLike


def clear_queue(q) -> None:
    print("Emptying queue.")
    try:
        while True:
            q.get(block=False)
    except queue.Empty:
        print("Queue emptied.")
    except ValueError:  # Queue is already closed
        print("Queue was already closed.")


@contextmanager
def open_tmsi_device(
    saga_config: str,
    verbose: bool = True,
) -> Generator[TMSiSDK.devices.saga.SagaDevice, None, None]:
    device = None
    try:
        print("Initializing TMSi device...")
        # Initialise the TMSi-SDK first before starting using it
        TMSiSDK.tmsi_device.initialize()
        # Execute a device discovery. This returns a list of device-objects.
        discovery_list = TMSiSDK.tmsi_device.discover(
            TMSiSDK.tmsi_device.DeviceType.saga,
            TMSiSDK.device.DeviceInterfaceType.docked,
            TMSiSDK.device.DeviceInterfaceType.usb,  # .network
        )
        if len(discovery_list) == 0:
            raise ValueError(
                "No TMSi device found. Please check your connections."
            )
        if len(discovery_list) > 1:
            raise ValueError(
                "More than one TMSi device found. Please check your"
                f" connections. Found: {discovery_list}."
            )
        # Get the handle to the first discovered device.
        device = discovery_list[0]
        print(f"Found device: {device}")
        device.open()
        print("Connected to device.")
        cfg_file = TMSiSDK.get_config(saga_config)
        device.load_config(cfg_file)
        if verbose:
            print("\nThe active channels are : ")
            for idx, ch in enumerate(device.channels):
                print(
                    "[{0}] : [{1}] in [{2}]".format(idx, ch.name, ch.unit_name)
                )
            print("\nCurrent device configuration:")
            print(
                f"Base-sample-rate: \t\t\t{device.config.base_sample_rate} Hz"
            )
            print(f"Sample-rate: \t\t\t\t{device.config.sample_rate} Hz")
            print(f"Reference Method: \t\t\t{device.config.reference_method}")
            print(
                f"Sync out configuration: \t{device.config.get_sync_out_config()}"
            )
        # TMSiSDK.devices.saga.xml_saga_config.xml_write_config(
        # filename=cfg_file, saga_config=device.config
        # )
        device.start_measurement()
        if device is None:
            raise ValueError("No TMSi device found!")
        yield device
    except TMSiSDK.error.TMSiError as error:
        print("!!! TMSiError !!! : ", error.code)
        if (
            device is not None
            and error.code == TMSiSDK.error.TMSiErrorCode.device_error
        ):
            print("  => device error : ", hex(device.status.error))
            TMSiSDK.error.DeviceErrorLookupTable(hex(device.status.error))
    except Exception as exception:
        if device is not None:
            if device.status.state == TMSiSDK.device.DeviceState.sampling:
                print("Stopping TMSi measurement...")
                device.stop_measurement()
            if device.status.state == TMSiSDK.device.DeviceState.connected:
                print("Closing TMSi device...")
                device.close()
        raise exception


@contextmanager
def open_lsl_stream(
    device,
) -> Generator[TMSiFileFormats.file_writer.FileWriter, None, None]:
    lsl_stream = TMSiFileFormats.file_writer.FileWriter(
        TMSiFileFormats.file_writer.FileFormat.lsl, "SAGA"
    )
    try:
        lsl_stream.open(device)
        yield lsl_stream
    except Exception as exception:
        print("Closing LSL stream...")
        lsl_stream.close()
        raise exception


@dataclass
class ProcessManager:
    device: TMSiSDK.devices.saga.SagaDevice
    lsl_stream: TMSiFileFormats.file_writer.FileWriter
    out_dir: _PathLike
    timeout: float = 0.05
    verbose: bool = True
    _terminated: bool = field(init=False, default=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._terminated:
            self.terminate()

    def __post_init__(self) -> None:
        self.out_dir = pathlib.Path(self.out_dir)
        self.queue_source = multiprocessing.Queue(
            int(self.timeout * 1000 * 20)
        )  # seconds/sample * ms/s * s
        self.queue_raw = multiprocessing.Queue(int(self.timeout * 1000))
        self.queue_features = multiprocessing.Queue(1)
        self.queue_decoding = multiprocessing.Queue(1)
        self.queues = [
            self.queue_raw,
            self.queue_features,
            self.queue_decoding,
            self.queue_source,
        ]
        for q in self.queues:
            q.cancel_join_thread()

    def start(self) -> None:
        def on_press(key) -> None:
            pass

        def on_release(key) -> Literal[False] | None:
            if key == Key.esc:
                print("Received stop key.")
                self.queue_source.put(None)
                self.terminate()
                return False

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        TMSiSDK.sample_data_server.registerConsumer(
            self.device.id, self.queue_source
        )
        features = realtime_decoding.Features(
            name="Features",
            source_id="features_1",
            n_feats=7,
            sfreq=self.device.config.sample_rate,
            interval=self.timeout,
            queue_raw=self.queue_source,
            queue_features=self.queue_features,
            out_dir=self.out_dir,
            path_grids=None,
            line_noise=50,
            verbose=self.verbose,
        )
        decoder = realtime_decoding.Decoder(
            queue_decoding=self.queue_decoding,
            queue_features=self.queue_features,
            interval=self.timeout,
            out_dir=self.out_dir,
            verbose=self.verbose,
        )
        processes = [features, decoder]
        for process in processes:
            process.start()
            time.sleep(0.5)

    def terminate(self) -> None:
        """Terminate all workers."""
        self._terminated = True
        self.queue_source.put(None)
        print("Set terminating event.")
        TMSiSDK.sample_data_server.unregisterConsumer(
            self.device.id, self.queue_source
        )
        print("Unregistered consumer.")

        self.lsl_stream.close()
        if self.device.status.state == TMSiSDK.device.DeviceState.sampling:
            self.device.stop_measurement()
            print("Controlled stopping TMSi measurement...")
        if self.device.status.state == TMSiSDK.device.DeviceState.connected:
            self.device.close()
            print("Controlled closing TMSi device...")

        # Check if all processes have terminated
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Wait for processes to temrinate on their own
        print(f"Alive processes: {list(p.name for p in active_children)}")
        print("Waiting for processes to finish. Please wait...")
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try flushing all queues
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Flushing all queues. Please wait...")
        for queue_ in self.queues:
            clear_queue(queue_)
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try killing all processes gracefully
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Trying to kill processes gracefully. Please wait...")
        interrupt = (
            signal.CTRL_C_EVENT if sys.platform == "win32" else signal.SIGINT
        )
        for process in active_children:
            if process.is_alive():
                os.kill(process.pid, interrupt)
        self.wait(active_children, timeout=5)
        active_children = multiprocessing.active_children()
        if not active_children:
            return

        # Try forcefully terminating processes
        print(f"Alive processes: {(p.name for p in active_children)}")
        print("Terminating processes forcefully.")
        for process in active_children:
            if process.is_alive():
                process.terminate()

    @staticmethod
    def wait(processes, timeout=None) -> None:
        """Wait for all workers to die."""
        if not processes:
            return
        start = time.time()
        while True:
            try:
                if all(not process.is_alive() for process in processes):
                    # All the workers are dead
                    return
                if timeout and time.time() - start >= timeout:
                    # Timeout
                    return
                time.sleep(0.1)
            except Exception:
                pass


def run(
    out_dir: _PathLike,
    saga_config: str = "saga_config_sensight_lfp_left",
) -> None:
    """Initialize data processing by launching all necessary processes."""
    with (
        open_tmsi_device(saga_config) as device,
        open_lsl_stream(device) as stream,
        ProcessManager(
            device=device,
            lsl_stream=stream,
            out_dir=out_dir,
            timeout=0.05,
            verbose=False,
        ) as manager,
    ):
        manager.start()


# if __name__ == "__main__":
#     stream_manager = run("saga_config_sensight_lfp_left")
