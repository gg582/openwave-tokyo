// ============================================================================
// texture_load.h — minimal JPEG loader (libjpeg) for CC0 material maps
// ============================================================================
#pragma once

// Loads a baseline JPEG as packed RGB8. Returns true on success and fills
// out/outW/outH. Caller frees `out` with delete[].
bool loadJPEG(const char* path, unsigned char** out, int* outW, int* outH);
