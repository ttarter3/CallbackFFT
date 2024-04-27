# . /data001/heterogene_mw/spack/share/spack/setup-env.sh -> ~/.bashrc
# ~/.bashrc

# spack load cuda@12.3


.PHONY: clean pgen run_p sgen run_s

run_p: 
	@./.runJob.sh p
run_s:
	@./.runJob.sh s

pgen:
	@mkdir -p ./install
	module load cuda && \
	cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --build build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --install build

sgen:
	@mkdir -p ./install
	cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --build build -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON && \
	cmake --install build

clean:
	rm -rf *.qsub_out
	cmake --build build --target clean