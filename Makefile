.DEFAULT_GOAL := all
.PHONY: all complete model1 model2 model3 pause

# all
all: clean complete
# pause model1 pause model3 pause model2

# pause
pause:
	@echo "Pause for 10 minutes"
	@sleep 600
	@echo ""

# remove old data
clean:
	@echo "Removing old data..."
	@rm -rf csv
	@rm -rf graphs
	@rm -rf logs
	@echo ""

# run complete network
complete:
	@echo "[GPU] Running the model complete..."
	@echo ""
	@python model_complete.py
	@echo ""

# run model 1
model1:
	@echo "[CPU] Running the model 1..."
	@echo ""
	@export CUDA_VISIBLE_DEVICES="" python model_no1.py
	@echo ""

# run model 2
model2:
	@echo "[CPU] Running the model 2..."
	@echo ""
	@export CUDA_VISIBLE_DEVICES="" python model_no2.py
	@echo ""

# run model 3
model3:
	@echo "[GPU] Running the model 3..."
	@python model_no3.py
	@echo ""