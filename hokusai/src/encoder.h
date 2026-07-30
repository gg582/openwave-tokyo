// ============================================================================
// encoder.h — H.264 MP4 encoder via the FFmpeg C API
//
// Preferred path: h264_nvenc (NVENC hardware encoder) fed with CUDA
// hw_frames. The RGBA->NV12 conversion kernel (postfx.cu) writes DIRECTLY
// into the AVFrame device planes, so pixels never leave the GPU:
//   GL PBO -> CUDA correction kernel -> NV12 planes (device) -> NVENC
// If NVENC is unavailable the class falls back to libx264 (CPU), keeping
// the same MP4/H.264 output contract.
// ============================================================================
#pragma once
#include <cstddef>

class Encoder {
public:
    bool open(const char* path, int w, int h, int fps);

    // --- NVENC path -------------------------------------------------------
    // Acquire the next hardware frame's device planes (fill them with the
    // rgbaToNv12 kernel, then call writeFrameDevice()).
    bool acquireDevicePlanes(unsigned char** y, int* yPitch,
                             unsigned char** uv, int* uvPitch);
    bool writeFrameDevice();

    // --- CPU fallback path -------------------------------------------------
    bool writeFrameHost(const unsigned char* rgba, int w, int h);

    void close();
    bool usingNvenc() const { return nvenc_; }

private:
    bool openCodec(const char* name, bool hw);
    bool drainPacket();
    bool encodeAndWrite();

    void* oc_ = nullptr;        // AVFormatContext
    void* cc_ = nullptr;        // AVCodecContext
    void* st_ = nullptr;        // AVStream
    void* hwDevice_ = nullptr;  // AVBufferRef (CUDA device)
    void* hwFrames_ = nullptr;  // AVBufferRef (frame pool)
    void* frame_ = nullptr;     // AVFrame
    void* pkt_ = nullptr;       // AVPacket
    long long pts_ = 0;
    int w_ = 0, h_ = 0, fps_ = 0;
    bool nvenc_ = false;
    bool headerWritten_ = false;
};
