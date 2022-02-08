#pragma once
#ifndef CONFIG_H

#ifndef NDEBUG // cmake defines NDEBUG instead of DEBUG
#define DEBUG
#endif

//#define DO_TERM_DISPLAY

#define BLOCK_SIZE 512

#ifndef NDIM
#define NDIM 2
#endif
#ifndef N_COLOR_BIT
//#define N_COLOR_BIT (sizeof(float))
#define N_COLOR_BIT 1L
#endif
#ifndef CANVAS_SIZE_X
//#define CANVAS_SIZE_X 32L
//#define CANVAS_SIZE_X 128L
//#define CANVAS_SIZE_X 500L
#define CANVAS_SIZE_X 1920L
//#define CANVAS_SIZE_X ((int)(1919 * 1.5))
//#define CANVAS_SIZE_X (ssize_t)99551
#endif
#ifndef CANVAS_SIZE_Y
//#define CANVAS_SIZE_Y 32L
//#define CANVAS_SIZE_Y 128L
//#define CANVAS_SIZE_Y 500L
#define CANVAS_SIZE_Y 1080L
//#define CANVAS_SIZE_Y ((int)(1080 * 1.5))
//#define CANVAS_SIZE_Y (ssize_t)99343
#endif

#ifndef NUM_RULE
#define NUM_RULE 10
#endif
#ifndef BUFFER_SIZE
#define BUFFER_SIZE ((CANVAS_SIZE_X * CANVAS_SIZE_Y + 1) * sizeof(cell))
#define NUM_CELLS (CANVAS_SIZE_X * CANVAS_SIZE_Y)
#endif

#define RATIO_CELL_TO_PIXEL 1
#define RENDER_WINDOW_WIDTH (CANVAS_SIZE_X * RATIO_CELL_TO_PIXEL)
#define RENDER_WINDOW_HEIGHT (CANVAS_SIZE_Y * RATIO_CELL_TO_PIXEL)
#define RENDER_WIDTH 0.95

#ifdef DEBUG
#define debug_print(...)     \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)
#else
#define debug_print(...) \
    do {                 \
    } while (0)
#endif
#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))

struct cell;
typedef struct cell {
    unsigned int x : N_COLOR_BIT;
    // float x;
} cell;
#endif
