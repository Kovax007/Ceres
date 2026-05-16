CXX = clang++
CXXFLAGS = -std=c++17 -fPIC -O3 -Wall -fvisibility=hidden -fstack-protector-strong -D_FORTIFY_SOURCE=2
CUDA_PATH ?= /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -shared -L$(CUDA_PATH)/lib64 -Wl,-z,relro,-z,now
LIBS = -lnvinfer -lnvonnxparser -lcudart

TARGET = libTensorRTWrapper.so
SOURCES = TensorRTWrapper.cpp
OBJECTS = $(SOURCES:.cpp=.o)

WORKER = trt_build_worker
WORKER_SOURCES = trt_build_worker.cpp
WORKER_OBJECTS = $(WORKER_SOURCES:.cpp=.o)

all: $(TARGET) $(WORKER)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(WORKER): $(WORKER_OBJECTS) $(TARGET)
	$(CXX) -o $@ $(WORKER_OBJECTS) -L. -lTensorRTWrapper -L$(CUDA_PATH)/lib64 -lcudart -Wl,-rpath,'$$ORIGIN'

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET) $(WORKER_OBJECTS) $(WORKER)

install: $(TARGET) $(WORKER)
	cp $(TARGET) ..
	cp $(WORKER) ..

.PHONY: all clean install
