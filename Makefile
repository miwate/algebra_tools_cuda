# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -Iinclude
ARCH = -arch=sm_75

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
BUILD_DIR = build

# Target
TARGET = $(BIN_DIR)/eigen_solver

# Source files
SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/kernels.cu $(SRC_DIR)/eigen_solver.cu
OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/kernels.o $(BUILD_DIR)/eigen_solver.o

# Create directories
$(shell mkdir -p $(BIN_DIR) $(BUILD_DIR))

# Build target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(ARCH) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CFLAGS) $(ARCH) -c -o $@ $<

# Clean
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)

# Run
run: $(TARGET)
	./$(TARGET)