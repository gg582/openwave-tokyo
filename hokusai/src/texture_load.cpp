// ============================================================================
// texture_load.cpp — minimal JPEG loader (libjpeg)
// ============================================================================
#include "texture_load.h"
#include <cstdio>
#include <cstring>
#include <jpeglib.h>

bool loadJPEG(const char* path, unsigned char** out, int* outW, int* outH)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        fclose(f);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    const int w = (int)cinfo.output_width;
    const int h = (int)cinfo.output_height;
    unsigned char* buf = new unsigned char[(size_t)w * h * 3];
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = buf + (size_t)cinfo.output_scanline * w * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);

    *out = buf; *outW = w; *outH = h;
    return true;
}
