import mne
import numpy as np

raw = mne.io.read_raw_edf("E:\Projects\EEG\dataset\Ezio_Solo_prog.edf", preload=True)

thalamus_electrodes = ['EEG T3-Cz', 'EEG T4-Cz', 'EEG T5-Cz', 'EEG T6-Cz']
sfreq = raw.info['sfreq']  # Sampling frequency

eyes_closed = raw.copy().crop(316, 630).pick_channels(ch_names=thalamus_electrodes)  # Data from eyes closed
experiment = raw.copy().crop(644, 3050).pick_channels(ch_names=thalamus_electrodes)  # Data from experiment
print(raw.find_edf_events())  # Printing events


# Performing FIR filtering to get only Alpha waves (8-12 Hz)
eyes_closed_filtered_data = mne.filter.filter_data(data=eyes_closed.get_data(), l_freq=8, h_freq=12,
                                                   sfreq=sfreq, method="fir")

experiment_filtered_data = mne.filter.filter_data(data=experiment.get_data(), l_freq=8, h_freq=12,
                                                  sfreq=sfreq, method="fir")

# Preparing data for plotting
eyes_closed_filtered = mne.io.RawArray(data=eyes_closed_filtered_data * 30,
                                       info=mne.create_info(ch_names=thalamus_electrodes, sfreq=sfreq))

experiment_filtered = mne.io.RawArray(data=experiment_filtered_data * 30,
                                      info=mne.create_info(ch_names=thalamus_electrodes, sfreq=sfreq))


eyes_closed_filtered.plot(block=True)
experiment_filtered.plot(block=True)

print("Mean value with eyes closed: " + str(np.mean(np.absolute(eyes_closed_filtered_data))))
print("Mean value on experiment: " + str(np.mean(np.absolute(experiment_filtered_data))))
print("Mean eyes closed to Mean experiment ratio: " +
      str(np.mean(np.absolute(eyes_closed_filtered_data))/np.mean(np.absolute(experiment_filtered_data))))