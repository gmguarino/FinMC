INC=-I/usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCC_OPT=-std=c++11 -lcurand

PYTHON=/usr/bin/python3

all:
	@echo "COMPILING SOURCE"
	@echo "--------------------------------------------------------"
	@$(NVCC) $(NVCC_OPT) montecarlo.cu mc_kernel.cu -o stonks.out
	@echo "Done\n"


run:
	@echo "Running MC Simulation"
	@./stonks.out

plot:
	@echo "Executing compiled program with plotting option"

	@./stonks.out plot
	@$(PYTHON) plot.py --save ./Images/hist.png
	@echo "Done\n"

clean:
	@echo "Removing Data"
	@echo "--------------------------------------------------------"
	@-rm -f stonks.out data.dat
	@echo "\nDone\n"

cleanplot:
	@echo "Removing Images and Data"
	@echo "--------------------------------------------------------"
	@-rm -f stonks.out ./Images/hist.png data.dat
	@echo "Done\n"