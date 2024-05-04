


import re

from matplotlib import pyplot as plt

data = dict()

def extract_numbers_from_file(filename):
    # Define a regular expression pattern to match numbers

    number_pattern = "Alg ([-+]?\d+): OperatingTime_millsec ([+-]?\d*\.\d+|\d) ([-+]?\d+)"
    legend = ['cuFFT', "Radix 2", "Radix 4", "Radix 8", "Radix2-SingleShared", "Radix2-2xThread", "Radix2-512Shared", "Radix2-Optimized"]

    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Iterate through each line
        for line in lines:
            # Split the line on "n:"
            parts = line.split("n:")
            # Iterate through each part
            for part in parts:
                # Find all numbers in the part using regular expression
                numbers = re.findall(number_pattern, part)
                # Print the numbers found in the part
                if numbers:
                    if numbers[0][0] in data:
                        data[numbers[0][0]]["time"].append(float(numbers[0][1]))
                        data[numbers[0][0]]["SIZE_BASE2_M"].append(int(numbers[0][2]))
                    else:
                        data[numbers[0][0]] = dict()
                        data[numbers[0][0]]["time"] = [float(numbers[0][1])]
                        data[numbers[0][0]]["SIZE_BASE2_M"] = [int(numbers[0][2])]

    plt.figure()
    for xx in data:
        plt.plot(data[xx]["SIZE_BASE2_M"], data[xx]["time"], label=legend[int(xx)])
    plt.xlabel("Size[2^M]")
    plt.ylabel("Time(milliseconds)")
    plt.legend()
    plt.show()



                # for number in numbers:

# Call the function with the file name
extract_numbers_from_file(r"C:\Users\ttart\PycharmProjects\Test\FFT\FinalProj.FFTTest.qsub_out")