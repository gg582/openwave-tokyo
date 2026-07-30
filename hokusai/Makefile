# ============================================================================
# Hokusai Wave — CUDA/OpenGL ocean simulation & rendering pipeline
#
# Two rendering paths share one scene:
#   traditional    : classic iterative radix-2 FFT + GLSL separable-Gaussian
#                    unsharp masking (control group)
#   complementary  : warp-shuffle antipodal-pair FFT + CUDA warp-shuffle
#                    inverse-OTF unsharp correction kernel
# ============================================================================

NVCC      := nvcc
CXX       := g++
ARCH      := -arch=native
CXXFLAGS  := -O3 -std=c++17 -Wall
NVCCFLAGS := -O3 -std=c++17 $(ARCH) --expt-relaxed-constexpr

GL_LIBS   := -lEGL -lGL
FFMPEG_LIBS := -lavformat -lavcodec -lavutil
CUDA_LIBS := -lcudart -lcuda -ljpeg
# libavformat may pull libsrt after the C++ runtime under CUDA's linker
# invocation. Keep the current system C++ ABI runtime explicitly after it.
LDLIBS := -Xlinker --no-as-needed -lstdc++ -Xlinker --allow-shlib-undefined

SRC_CPP   := src/main.cpp src/bathymetry.cpp src/renderer.cpp src/encoder.cpp src/texture_load.cpp src/terrain.cpp src/climate.cpp src/tide.cpp
SRC_CU    := src/fft_gpu.cu src/ocean.cu src/postfx.cu src/spray.cu

BIN       := hokusai_wave

all: $(BIN)

$(BIN): $(SRC_CPP) $(SRC_CU) src/*.h
	$(NVCC) $(NVCCFLAGS) -x cu $(SRC_CPP) $(SRC_CU) -o $@ \
	    $(GL_LIBS) $(FFMPEG_LIBS) $(CUDA_LIBS) $(LDLIBS)

clean:
	rm -f $(BIN) *.mp4

.PHONY: all clean
