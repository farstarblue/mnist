CC := cc
CFLAGS := -std=c99 -O3 -Wall -Wextra -pedantic
LDFLAGS := -lm

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
SCRIPT_DIR := scripts

TRAIN_BIN := $(BIN_DIR)/train
VERIFY_BIN := $(BIN_DIR)/verify

COMMON_OBJS := $(BUILD_DIR)/mnist.o $(BUILD_DIR)/network.o
TRAIN_OBJS := $(COMMON_OBJS) $(BUILD_DIR)/model_io.o $(BUILD_DIR)/train.o
VERIFY_OBJS := $(COMMON_OBJS) $(BUILD_DIR)/verify.o

ARGS ?=

.PHONY: all data train verify clean

all: $(TRAIN_BIN) $(VERIFY_BIN)

data:
	bash $(SCRIPT_DIR)/download_mnist.sh

train: $(TRAIN_BIN) data
	./$(TRAIN_BIN) $(ARGS)
	@$(MAKE) --no-print-directory $(VERIFY_BIN)

verify: $(VERIFY_BIN) data
	./$(VERIFY_BIN) $(ARGS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(TRAIN_BIN) $(VERIFY_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/mnist.o: $(SRC_DIR)/mnist.c $(SRC_DIR)/mnist.h $(SRC_DIR)/config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/network.o: $(SRC_DIR)/network.c $(SRC_DIR)/network.h $(SRC_DIR)/config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/model_io.o: $(SRC_DIR)/model_io.c $(SRC_DIR)/model_io.h $(SRC_DIR)/network.h $(SRC_DIR)/config.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/train.o: $(SRC_DIR)/train.c $(SRC_DIR)/config.h $(SRC_DIR)/mnist.h $(SRC_DIR)/model_io.h $(SRC_DIR)/network.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/verify.o: $(SRC_DIR)/verify.c $(SRC_DIR)/config.h $(SRC_DIR)/mnist.h $(SRC_DIR)/model_params.h $(SRC_DIR)/network.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TRAIN_BIN): $(TRAIN_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(TRAIN_OBJS) -o $@ $(LDFLAGS)

$(VERIFY_BIN): $(VERIFY_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(VERIFY_OBJS) -o $@ $(LDFLAGS)
