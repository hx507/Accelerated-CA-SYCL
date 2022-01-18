#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <CL/sycl.hpp>

#include "config.h"

namespace sycl = cl::sycl;
sycl::queue cl_queue(sycl::default_selector {});

typedef sycl::buffer<cell, 1> cl_buffer;
static cell *canvas1 = (cell*)calloc(BUFFER_SIZE, 1), *canvas2 = (cell*)calloc(BUFFER_SIZE, 1);
cl_buffer buffer1 { canvas1, CANVAS_SIZE_X*CANVAS_SIZE_Y }, buffer2 { canvas2, CANVAS_SIZE_X*CANVAS_SIZE_Y };

cell* host_canvas = (cell*)calloc(BUFFER_SIZE, 1);
cl_buffer host_buffer { host_canvas, CANVAS_SIZE_X* CANVAS_SIZE_Y };

inline size_t idx(ssize_t X, ssize_t Y)
{
    if (Y >= 0 && X >= 0 && ((Y * CANVAS_SIZE_X + X) < (long)(NUM_CELLS - 1))) {
        return (Y * CANVAS_SIZE_X + X);
    } else {
        return NUM_CELLS - 1;
    }
}
inline size_t xdi_x(size_t i)
{
    return i % CANVAS_SIZE_X;
}

inline size_t xdi_y(size_t i)
{
    return i / CANVAS_SIZE_X;
}

inline unsigned int update_cell_bin_2d(const cell& center, const cell& c1, const cell& c2, const cell& c3, const cell& c4, const cell& c5, const cell& c6, const cell& c7, const cell& c8)
{
    unsigned int sum = c1.x
        + c2.x
        + c3.x
        + c4.x
        + c5.x
        + c6.x
        + c7.x
        + c8.x;
    return (center.x && (sum == 2 || sum == 3)) || ((!center.x) && sum == 3);
}

inline void update(cl_buffer origin, cl_buffer dest, sycl::queue cl_queue)
{
    {
        cl_queue.submit([&](sycl::handler& cgh) {
            auto ori_acc = origin.get_access<sycl::access::mode::read>(cgh);
            auto dst_acc = dest.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for(sycl::range<1>(CANVAS_SIZE_X * CANVAS_SIZE_Y), [=](sycl::id<1> i) {
                auto x = xdi_x(i), y = xdi_y(i);
                dst_acc[i].x = update_cell_bin_2d(ori_acc[i],
                    ori_acc[idx(x - 1, y - 1)],
                    ori_acc[idx(x + 1, y + 1)],
                    ori_acc[idx(x, y - 1)],
                    ori_acc[idx(x - 1, y)],
                    ori_acc[idx(x + 1, y)],
                    ori_acc[idx(x, y + 1)],
                    ori_acc[idx(x - 1, y + 1)],
                    ori_acc[idx(x + 1, y - 1)]);
            });
        });
    }
}

inline void buf_copy(cl_buffer src, cl_buffer dst)
{
    cl_queue.submit([&](sycl::handler& cgh) {
        auto src_a = src.get_access<sycl::access::mode::read>(cgh);
        auto dst_a = dst.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.copy(src_a, dst_a);
    });
}

inline void copy_into_buffer(cell* src, cl_buffer dst_buf)
{
    auto dst = dst_buf.get_access<sycl::access::mode::discard_write>();
    for (size_t i = 0; i < CANVAS_SIZE_X * CANVAS_SIZE_Y; i++) {
        dst[i] = src[i];
    }
}
inline void copy_from_buffer(cl_buffer src_buf, cell* dst)
{
    auto src = src_buf.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < CANVAS_SIZE_X * CANVAS_SIZE_Y; i++) {
        dst[i] = src[i];
    }
}

inline void print_buffer(cl_buffer src_buf)
{
#ifdef DO_TERM_DISPLAY
    auto src = src_buf.get_access<sycl::access::mode::read>();
    fprintf(stdout, "---------------------Iteration-------------------------\n");
    for (ssize_t x = 0; x < CANVAS_SIZE_X; x++) {
        for (ssize_t y = 0; y < CANVAS_SIZE_Y; y++) {
            // debug_print("(%zu, %zu) -> %zu\n", x, y, idx(x, y));
            fprintf(stdout, src[idx(x, y)].x ? " " : "█");
        }
        fprintf(stdout, "|\n");
    }
#endif
}

int main(int argc, char** argv)
{
    debug_print("-------------CA Running-----------\n");
    debug_print("Using %d dimensional canvas of size %zux%zu with %lu bit colors\n", NDIM, CANVAS_SIZE_X, CANVAS_SIZE_Y, N_COLOR_BIT);
    debug_print("Using two buffer each of size %lu mb, or %lu cells\n", BUFFER_SIZE / 1024 / 1024, BUFFER_SIZE / sizeof(cell));

    std::cout << "Running on "
              << cl_queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    for (int i = CANVAS_SIZE_X * 3 / 7; i < CANVAS_SIZE_X * 4 / 7; i++) {
        host_canvas[idx(i, i)].x = 1;
        host_canvas[idx(i, i + 1)].x = 1;
        host_canvas[idx(i, i - 1)].x = 1;
        host_canvas[idx(i, i - 5)].x = 1;
        host_canvas[idx(i - 1, i - 6)].x = 1;
        host_canvas[idx(i, i - 6)].x = 1;
    }
    copy_into_buffer(host_canvas, buffer1);
    print_buffer(buffer1);

    int iteration = 100000;
#ifdef DO_TERM_DISPLAY
    int delay = 30000;
#else
    // int delay = 10000;
    int delay = 0;
#endif
    debug_print("Display is now ready\n");
    for (int i = 0; i < iteration / 2; i++) {
        update(buffer1, buffer2, cl_queue);
        // copy_from_buffer(buffer2, host_canvas);
        print_buffer(buffer2);
        usleep(delay);

        update(buffer2, buffer1, cl_queue);
        // copy_from_buffer(buffer2, host_canvas);
        print_buffer(buffer1);
        usleep(delay);
    }
}
