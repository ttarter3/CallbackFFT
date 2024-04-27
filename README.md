
# Repository 

https://github.com/ttarter3/CallbackFFT.git

# Build and Run

to build and run the code on a ASA-X type
```
git submodulate update --init

make sgen
make run_s
```

to build and run the code on Voyager1 type

```
# Make sure to run:
. /data001/heterogene_mw/spack/share/spack/setup-env.sh -> ~/.bashrc
```

```
git submodulate update --init

make pgen
make run_p
```

# Comparison of Implmentations
assuming python3 is installed, you can run the following to see the plot of the generated data in terms of SNR:
```
cd ./python_scripts
./runPlotFFT.sh
```

The code compares the calculated values to cuFFT.  If the average error is greater than the threshold, then we write out a error message to the screen.  Either "Error: Invalid Vector Size" or "Error: Measurment Error Greater than expected(###)".  There are also data checks to make sure that the algorithm you are requesting to use can support the size of the data being used. This error message looks something like "Error: Radix # must satisfy 0 == log2(M)"