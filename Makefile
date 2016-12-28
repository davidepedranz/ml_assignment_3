.DEFAULT_GOAL := all
.PHONY: all complete model1 model2 model3

# export CUDA_VISIBLE_DEVICES=""

# all
all: clean complete model1 model2 model3

# remove old data
clean:
	@echo "Removing old data..."
	@rm -rf csv
	@rm -rf graphs
	@rm -rf logs
	@echo ""

# run complete network
complete:
	@echo "Running the model complete..."
	@echo ""
	@python model_complete.py
	@echo ""

# run model 1
model1:
	@echo "Running the model 1..."
	@echo ""
	@python model_no1.py
	@echo ""

# run model 2
model2:
	@echo "Running the model 2..."
	@echo ""
	@python model_no2.py
	@echo ""

# run model 3
model3:
	@echo "Running the model 3..."
	@python model_no3.py
	@echo ""