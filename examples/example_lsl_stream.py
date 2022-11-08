from audioop import mul
from lib2to3.pytree import Base
import pandas as pd
import os
import multiprocessing


from threading import Timer
import numpy as np
import pylsl
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from pynput.keyboard import Key, Listener
from py_neuromodulation import (
    nm_lsl_stream,
    nm_define_nmchannels,
    nm_RealTimeStreamApp,
    nm_stream_abc,
)


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def setup_stream():
    ch_names = [
        "FP1",
        "FP2",
        "C3",
        "C4",
        "P7",
        "P8",
        "O1",
        "O2",
    ]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=ch_names,
        ch_types=["eeg" for _ in range(len(ch_names))],
        reference=["average" for _ in range(len(ch_names))],
        bads=None,
        used_types=("eeg",),
    )

    # set used to false for all channels except C3 and C4
    nm_channels.used = [0, 0, 1, 1, 0, 0, 0, 0]
    stream = nm_lsl_stream.LSLStream(
        settings=None,
        nm_channels=nm_channels,
        path_grids=None,
        verbose=True,
    )

    for f in stream.settings["features"]:
        stream.settings["features"][f] = False
    stream.settings["features"]["fft"] = True

    for f in stream.settings["preprocessing"]:
        stream.settings["preprocessing"][f] = False
    stream.settings["preprocessing"]["re_referencing"] = True
    stream.settings["preprocessing"]["notch_filter"] = True
    stream.settings["preprocessing"]["preprocessing_order"] = [
        "re_referencing",
        "notch_filter",
    ]

    for f in stream.settings["postprocessing"]:
        stream.settings["postprocessing"][f] = False
    stream.settings["postprocessing"]["feature_normalization"] = True
    stream.settings["feature_normalization_settings"][
        "normalization_method"
    ] = "zscore"

    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [60, 80],
    }
    stream.settings["sampling_rate_features_hz"] = 1
    stream.init_stream(
        sfreq=250,
        line_noise=50,
        coord_list=None,
        coord_names=None,
    )

    return stream


def main():

    stream_ = setup_stream()

    app = nm_RealTimeStreamApp.StreamApp(
        stream_,
        VERBOSE=False,
        TRAINING=False,
        PREDICTION=True,
        training_samples_each_cond_s=30,
        PATH_OUT="/Users/hi/Documents/py_neuromodulation/examples/model_real_time_train",
    )

    # pylsl needs to be started from the main thread
    # therefore the get_data function cannot be set in nm_stream
    streams = pylsl.resolve_streams(
        wait_time=1,
    )

    lsl_streaminlet = pylsl.StreamInlet(info=streams[0], max_buflen=50)

    def get_data(queue_raw: multiprocessing.Queue):
        samples, _ = lsl_streaminlet.pull_chunk(max_samples=50, timeout=1)
        raw_data = np.array(samples).T  # shape (ch, time)
        queue_raw.put(raw_data)

    # the KeyListener is also a Thread object
    def on_press(key):
        print("{0} pressed".format(key))

    def on_release(key):
        if key == Key.caps_lock:
            # Stop listener
            print("reiceived stop key pressed")

            app.queue_raw.put(None)
            timer.cancel()
            return False

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    processes = [
        multiprocessing.Process(
            target=app.get_features_wrapper,
            args=(
                app.queue_raw,
                app.queue_features,
            ),
        ),
        multiprocessing.Process(
            target=app.process_features,
            args=(
                app.queue_features,
                app.queue_plotting,
            ),
        ),
    ]

    for p in processes:
        p.start()

    time_call_get_data_s = np.round(
        1 / app.stream.settings["sampling_rate_features_hz"], 2
    )
    time_call_get_data_s = 1  # in seconds

    # RepeatTimer will allow synchronized data pull
    timer = RepeatTimer(time_call_get_data_s, get_data, args=(app.queue_raw,))
    timer.start()

    def update(
        frame, queue_plotting: multiprocessing.Queue
    ):  # here frame needs to be accepted by the function since this is used in FuncAnimations
        data = queue_plotting.get()  # this blocks untill it gets some data
        for idx, rect in enumerate(bar_plt):
            rect.set_height(data[idx])
        plt.pause(0.005)
        return bar_plt

    if app.PREDICTION is True:
        fig, ax = plt.subplots()
        bar_plt = plt.bar([0, 1, 2], [0, 1, 2])
        plt.xticks([0, 1, 2], ["left", "rest", "right"])

        ani = FuncAnimation(
            fig, update, blit=False, fargs=(app.queue_plotting,)
        )
        plt.show()

    for p in processes:
        p.join()


if __name__ == "__main__":

    main()
