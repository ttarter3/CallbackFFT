import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt

def SearchFiles(directory, pattern):
    """
    Search a directory for files matching a pattern.
    Args:
        directory (str): Directory to search.
        pattern (str): Regular expression pattern to match filenames.
    Returns:
        list: List of filenames matching the pattern.
    """
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if re.match(pattern, file):
                matched_files.append(os.path.join(root, file))
    return matched_files

def ParseBinaryFile(filename, dtype):
    """
    Parse a binary data file using NumPy.
    Args:
        filename (str): Path to the binary data file.
    Returns:
        numpy.ndarray: Numpy array containing the parsed data.
    """
    with open(filename, 'rb') as f:
        # Adjust the dtype and shape according to your binary file structure
        # Here, assuming the data is stored as float32 values in a flat array
        data = f.read()
        data_size = np.frombuffer(data[0:8], dtype=np.uint64)
        data = np.frombuffer(data[8:], dtype=dtype)
    return data, data_size

if __name__ == "__main__":
    parsed_files = dict()
    # Directory to search for files
    search_directory = "../data"

    config_pattern = r'Config\.(\d+)\.bin$'
    fft_pattern = r'FreqData\.(\d+)\.bin$'

    legend = ['cuFFT', "Radix 2", "Radix 4", "Radix 8", "Radix2-SingleShared", "Radix2-2xThread", "Radix2-512Shared", "Radix2-Optimized"]

    cnt = 0
    while True:
        fig = None

        cnt = cnt + 1
        # Search for files matching the pattern
        files = SearchFiles(search_directory, fft_pattern)

        if cnt % 10 == 0:
            print('.', end='', flush=True)

        if files:
            files = SearchFiles(search_directory, config_pattern)
            config_data, d_size = ParseBinaryFile(files[0], np.float32)
            fs = config_data[0]

            fft_data_files = sorted(SearchFiles(search_directory, fft_pattern))

            if fft_data_files:
                for idx, file in enumerate(fft_data_files):
                    parsed_data, d_size = ParseBinaryFile(file, np.csingle)
                    if file not in parsed_files or not np.array_equal(parsed_data, parsed_files[file]):
                        print(f"Parsing file {cnt}: {file}")
                        parsed_files[file] = parsed_data

                        if fig is None:
                            fig = plt.figure()
                if fig is not None:
                    for idx, file in enumerate(fft_data_files):
                        freq = np.linspace(-fs / 2, fs / 2, parsed_files[file].size)

                        signal = parsed_files[file]
                        plottable_data = 10 * np.log10(np.abs(signal))
                        snr_data =  plottable_data#  - np.percentile(plottable_data, 95)

                        FLOOR = -40
                        VALID = FLOOR < snr_data
                        freq = freq[VALID]
                        snr_data = snr_data[VALID]


                        number_match = re.search(fft_pattern, file)
                        extracted_number = number_match.group(1)

                        if len(snr_data) != 0:
                            plt.plot(freq, snr_data, linewidth=3 * (len(fft_data_files) - idx), label=legend[int(extracted_number)])
                        else: print("No SNR Values")

                    plt.xlabel("Frequency[Hz]")
                    plt.ylabel("Amplitude[DB10]")
                    plt.title(f"SNR fs: {fs}\nExpectSig: {config_data[1:]}")
                    plt.ylim(bottom=FLOOR)  # Adjust the value as needed
                    plt.legend()
                    plt.show()
            time.sleep(1)

        else:
            print("No files matching the pattern found in the directory.")
