CXX = clang++
CXXFLAGS = -std=c++17 -fPIC -O3 -Wall -fvisibility=hidden -fstack-protector-strong -D_FORTIFY_SOURCE=2
CUDA_PATH = /usr/local/cuda-12.5
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -shared -L$(CUDA_PATH)/lib64 -Wl,-z,relro,-z,now
LIBS = -lnvinfer -lnvonnxparser -lcudart

TARGET = libTensorRTWrapper.so
SOURCES = TensorRTWrapper.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)

install: $(TARGET)
	cp $(TARGET) ..

.PHONY: all clean install
