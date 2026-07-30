// ============================================================================
// encoder.cpp — FFmpeg C API: NVENC (CUDA hw_frames) with libx264 fallback
// ============================================================================
#include "encoder.h"
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

bool Encoder::openCodec(const char* name, bool hw)
{
    const AVCodec* codec = avcodec_find_encoder_by_name(name);
    if (!codec) { fprintf(stderr, "encoder %s not found\n", name); return false; }

    AVCodecContext* cc = avcodec_alloc_context3(codec);
    cc->width      = w_;
    cc->height     = h_;
    cc->time_base  = AVRational{ 1, fps_ };
    cc->framerate  = AVRational{ fps_, 1 };
    cc->gop_size   = fps_;              // GOP 1 sec for fast seeking & stream compatibility
    cc->max_b_frames = 0;
    cc->bit_rate   = 12000000;          // 12 Mbps for 540p 60s stream

    AVDictionary* opt = nullptr;
    if (hw) {
        // CUDA frame pool: NV12 sw format is what NVENC consumes
        if (av_hwdevice_ctx_create((AVBufferRef**)&hwDevice_,
                                   AV_HWDEVICE_TYPE_CUDA, nullptr,
                                   nullptr, 0) < 0) {
            fprintf(stderr, "CUDA hwdevice creation failed\n");
            avcodec_free_context(&cc);
            return false;
        }
        AVBufferRef* framesRef = av_hwframe_ctx_alloc((AVBufferRef*)hwDevice_);
        AVHWFramesContext* fc = (AVHWFramesContext*)framesRef->data;
        fc->format    = AV_PIX_FMT_CUDA;
        fc->sw_format = AV_PIX_FMT_NV12;
        fc->width     = w_;
        fc->height    = h_;
        fc->initial_pool_size = 8;
        if (av_hwframe_ctx_init(framesRef) < 0) {
            fprintf(stderr, "hwframe ctx init failed\n");
            av_buffer_unref(&framesRef);
            avcodec_free_context(&cc);
            return false;
        }
        cc->pix_fmt       = AV_PIX_FMT_CUDA;
        cc->hw_frames_ctx = framesRef;   // ownership moves to cc
        av_dict_set(&opt, "preset", "p4", 0);
        av_dict_set(&opt, "tune", "hq", 0);
        av_dict_set(&opt, "rc", "vbr", 0);
        av_dict_set(&opt, "cq", "20", 0);
    } else {
        cc->pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opt, "preset", "medium", 0);
        av_dict_set(&opt, "crf", "18", 0);
    }

    const int ret = avcodec_open2(cc, codec, &opt);
    av_dict_free(&opt);
    if (ret < 0) {
        char err[128];
        av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "avcodec_open2(%s) failed: %s\n", name, err);
        avcodec_free_context(&cc);
        return false;
    }
    cc_ = cc;
    return true;
}

bool Encoder::open(const char* path, int w, int h, int fps)
{
    w_ = w; h_ = h; fps_ = fps;

    AVFormatContext* oc = nullptr;
    if (avformat_alloc_output_context2(&oc, nullptr, "mp4", path) < 0 || !oc) {
        fprintf(stderr, "cannot allocate mp4 output context\n");
        return false;
    }
    oc_ = oc;

    // try NVENC first, fall back to libx264
    nvenc_ = openCodec("h264_nvenc", true);
    if (!nvenc_) {
        fprintf(stderr, "NVENC unavailable, falling back to libx264\n");
        if (hwDevice_) { av_buffer_unref((AVBufferRef**)&hwDevice_); }
        if (!openCodec("libx264", false)) return false;
    }

    AVCodecContext* cc = (AVCodecContext*)cc_;
    AVStream* st = avformat_new_stream(oc, nullptr);
    st->time_base = cc->time_base;
    st->avg_frame_rate = cc->framerate;
    st->r_frame_rate = cc->framerate;
    avcodec_parameters_from_context(st->codecpar, cc);
    st_ = st;

    if (avio_open(&oc->pb, path, AVIO_FLAG_WRITE) < 0) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    AVDictionary* mopt = nullptr;
    av_dict_set(&mopt, "movflags", "+faststart", 0);
    if (avformat_write_header(oc, &mopt) < 0) {
        fprintf(stderr, "write_header failed\n");
        av_dict_free(&mopt);
        return false;
    }
    av_dict_free(&mopt);
    headerWritten_ = true;

    frame_ = av_frame_alloc();
    pkt_   = av_packet_alloc();
    return frame_ && pkt_;
}

bool Encoder::acquireDevicePlanes(unsigned char** y, int* yPitch,
                                  unsigned char** uv, int* uvPitch)
{
    if (!nvenc_) return false;
    AVFrame* f = (AVFrame*)frame_;
    av_frame_unref(f);
    // the hw frame pool lives on the codec context after avcodec_open2
    AVBufferRef* pool = ((AVCodecContext*)cc_)->hw_frames_ctx;
    if (av_hwframe_get_buffer(pool, f, 0) < 0)
        return false;
    *y      = f->data[0];
    *yPitch = f->linesize[0];
    *uv     = f->data[1];
    *uvPitch= f->linesize[1];
    return true;
}

bool Encoder::encodeAndWrite()
{
    AVCodecContext* cc = (AVCodecContext*)cc_;
    AVPacket* pkt = (AVPacket*)pkt_;
    AVStream* st = (AVStream*)st_;
    AVFormatContext* oc = (AVFormatContext*)oc_;

    if (avcodec_send_frame(cc, (AVFrame*)frame_) < 0) return false;
    while (avcodec_receive_packet(cc, pkt) == 0) {
        av_packet_rescale_ts(pkt, cc->time_base, st->time_base);
        pkt->stream_index = st->index;
        if (av_interleaved_write_frame(oc, pkt) < 0) return false;
        av_packet_unref(pkt);
    }
    return true;
}

bool Encoder::writeFrameDevice()
{
    AVFrame* f = (AVFrame*)frame_;
    f->pts = pts_++;
    return encodeAndWrite();
}

bool Encoder::writeFrameHost(const unsigned char* rgba, int w, int h)
{
    AVFrame* f = (AVFrame*)frame_;
    av_frame_unref(f);
    f->format = AV_PIX_FMT_YUV420P;
    f->width  = w;
    f->height = h;
    if (av_frame_get_buffer(f, 32) < 0) return false;
    // CPU RGBA -> I420 (BT.709 limited, matches the CUDA kernel);
    // glReadPixels rows are bottom-up, so flip vertically here.
    for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx) {
            const unsigned char* p = rgba + ((size_t)(h - 1 - yy) * w + xx) * 4;
            const float R = p[0], G = p[1], B = p[2];
            const float Y = 16.f + 0.2126f * R + 0.7152f * G + 0.0722f * B;
            f->data[0][yy * f->linesize[0] + xx] =
                (unsigned char)(Y < 0 ? 0 : Y > 255 ? 255 : Y);
            if ((xx & 1) == 0 && (yy & 1) == 0) {
                const float U = 128.f - 0.1146f * R - 0.3854f * G + 0.5f * B;
                const float V = 128.f + 0.5f * R - 0.4542f * G - 0.0458f * B;
                f->data[1][(yy / 2) * f->linesize[1] + xx / 2] =
                    (unsigned char)(U < 0 ? 0 : U > 255 ? 255 : U);
                f->data[2][(yy / 2) * f->linesize[2] + xx / 2] =
                    (unsigned char)(V < 0 ? 0 : V > 255 ? 255 : V);
            }
        }
    f->pts = pts_++;
    return encodeAndWrite();
}

bool Encoder::drainPacket()
{
    AVCodecContext* cc = (AVCodecContext*)cc_;
    AVPacket* pkt = (AVPacket*)pkt_;
    AVStream* st = (AVStream*)st_;
    AVFormatContext* oc = (AVFormatContext*)oc_;
    avcodec_send_frame(cc, nullptr);
    while (avcodec_receive_packet(cc, pkt) == 0) {
        av_packet_rescale_ts(pkt, cc->time_base, st->time_base);
        pkt->stream_index = st->index;
        av_interleaved_write_frame(oc, pkt);
        av_packet_unref(pkt);
    }
    return true;
}

void Encoder::close()
{
    if (!oc_) return;
    drainPacket();
    av_write_trailer((AVFormatContext*)oc_);
    if (frame_) av_frame_free((AVFrame**)&frame_);
    if (pkt_)   av_packet_free((AVPacket**)&pkt_);
    if (cc_)    avcodec_free_context((AVCodecContext**)&cc_);
    if (hwDevice_) av_buffer_unref((AVBufferRef**)&hwDevice_);
    AVFormatContext* oc = (AVFormatContext*)oc_;
    if (oc->pb) avio_closep(&oc->pb);
    avformat_free_context(oc);
    oc_ = nullptr;
}
