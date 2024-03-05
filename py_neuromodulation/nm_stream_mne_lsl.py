from mne_lsl.player import PlayerLSL
from mne_lsl.stream import StreamLSL
from mne_lsl import stream_viewer
from matplotlib import pyplot as plt
import numpy as np
import threading


if __name__ == "__main__":

    f_name = r"C:\code\py_neuromodulation\py_neuromodulation\data\sub-testsub\ses-EphysMedOff\ieeg\sub-testsub_ses-EphysMedOff_task-gripforce_run-0_ieeg.vhdr"
    raw = mne.io.read_raw_brainvision(f_name)
    # plt.figure()
    # plt.plot([1, 4, 5])
    # plt.show(block=True)

    player = PlayerLSL(f_name, name="example_stream", chunk_size=100)
    player = player.start()

    player.info

    sfreq = player.info["sfreq"]

    chunk_size = player.chunk_size
    interval = chunk_size / sfreq  # in seconds
    print(f"Interval between 2 push operations: {interval} seconds.")

    stream = StreamLSL(name="example_stream", bufsize=2).connect()
    ch_types = stream.get_channel_types(unique=True)
    print(f"Channel types included: {', '.join(ch_types)}")

    # viewer = stream_viewer.StreamViewer(stream_name="example_stream")
    # viewer.start()

    data_l = []
    timestamps_l = []
    idx_ = 0

    # start = time.time()
    def call_every_100ms():
        data, timestamps = stream.get_data(winsize=100)
        print(data.shape)
        data_l.append(data)
        timestamps_l.append(timestamps)

    t = threading.Timer(0.1, call_every_100ms)
    t.start()

    import time

    time_start = time.time()

    while time.time() - time_start <= 10:
        time.sleep(1)
    t.cancel()

    #    while idx_ < 100:
    #        if stream.n_new_samples >= 100:

    # now = time.time()
    # if now - start >= 0.1:
    #    #do_stuff()
    #    start = now

    # time.sleep(1)

    # plt.figure()
    # plt.plot(timestamps)
    # plt.show(block=True)
    # check here 1. the timing
    print(np.concatenate(data_l).shape)

    stream.disconnect()
    player.stop()
