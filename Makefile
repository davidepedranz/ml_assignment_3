.DEFAULT_GOAL := default
.PHONY: default complete model1 model2 model3 pause

default:
	@echo "Please select a task..."
	@echo ""

run_all: clean complete pause model1 pause model3 pause model2
	@echo "Executed all the models!"
	@echo ""

pause:
	@echo "Pause for 5 minutes"
	@sleep 300
	@echo ""

clean:
	@echo "Removing old data..."
	@rm -rf csv
	@rm -rf graphs
	@rm -rf logs
	@echo ""

complete:
	@echo "[GPU] Running the model complete..."
	@echo ""
	@python model_complete.py
	@echo ""

model1:
	@echo "[CPU] Running the model 1..."
	@echo ""
	@CUDA_VISIBLE_DEVICES="" python model_no1.py
	@echo ""

model2:
	@echo "[CPU] Running the model 2..."
	@echo ""
	@CUDA_VISIBLE_DEVICES="" python model_no2.py
	@echo ""

model3:
	@echo "[GPU] Running the model 3..."
	@python model_no3.py
	@echo ""
