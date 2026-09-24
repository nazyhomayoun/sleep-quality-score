import mne
import matplotlib.pyplot as plt

file_path = r"sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"

print("Loading EDF file...")

raw = mne.io.read_raw_edf(
    file_path,
    preload=True
)

print("\n========== BASIC INFORMATION ==========")

print("Channels:")
for i, channel in enumerate(raw.ch_names):
    print(f"{i}: {channel}")

print("\nSampling frequency:")
print(raw.info["sfreq"], "Hz")

print("\nDuration:")
print(raw.times[-1], "seconds")
print(raw.times[-1] / 3600, "hours")

print("\n=======================================")

# فقط 30 ثانیه اول
duration = 30

# گرفتن داده‌های 30 ثانیه اول
data = raw.get_data(
    start=0,
    stop=int(duration * raw.info["sfreq"])
)

times = raw.times[:data.shape[1]]

# Plot جداگانه برای هر کانال
for i, channel_name in enumerate(raw.ch_names):

    plt.figure(figsize=(14, 4))

    plt.plot(times, data[i])

    plt.title(channel_name)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.tight_layout()

    plt.show()