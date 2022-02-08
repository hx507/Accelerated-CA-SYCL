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

cell* canvas1 = sycl::malloc_device<cell>(BUFFER_SIZE, cl_queue);
cell* canvas2 = sycl::malloc_device<cell>(BUFFER_SIZE, cl_queue);

auto* host_canvas = new cell[NUM_CELLS];

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

inline void update(cell* origin, cell* dest, sycl::queue cl_queue)
{
    cl_queue.parallel_for(sycl::range<1>(CANVAS_SIZE_X * CANVAS_SIZE_Y), [=](sycl::id<1> i) {
        auto x = xdi_x(i), y = xdi_y(i);
        dest[i].x = update_cell_bin_2d(origin[i],
            origin[idx(x - 1, y - 1)],
            origin[idx(x + 1, y + 1)],
            origin[idx(x, y - 1)],
            origin[idx(x - 1, y)],
            origin[idx(x + 1, y)],
            origin[idx(x, y + 1)],
            origin[idx(x - 1, y + 1)],
            origin[idx(x + 1, y - 1)]);
    });
}

inline void memCopyHostToDevice(cell* host, cell* device)
{
    cl_queue.memcpy(device, host, BUFFER_SIZE);
}

inline void memCopyDeviceToHost(cell* device, cell* host)
{
    cl_queue.memcpy(host, device, BUFFER_SIZE);
}

inline void print_buffer(cell* src)
{
#ifdef DO_TERM_DISPLAY
    fprintf(stdout, "---------------------Iteration-------------------------\n");
    for (ssize_t x = 0; x < CANVAS_SIZE_X; x++) {
        for (ssize_t y = 0; y < CANVAS_SIZE_Y; y++) {
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
    memCopyHostToDevice(host_canvas, canvas1);
    print_buffer(host_canvas);

    int iteration = 10000;
#ifdef DO_TERM_DISPLAY
    int delay = 30000;
#else
    // int delay = 10000;
    int delay = 0;
#endif

    for (int i = 0; i < iteration / 2; i++) {
        update(canvas1, canvas2, cl_queue);
        memCopyDeviceToHost(canvas2, host_canvas);
        print_buffer(canvas2);
        usleep(delay);

        cl_queue.wait_and_throw();

        update(canvas2, canvas1, cl_queue);
        memCopyDeviceToHost(canvas2, host_canvas);
        print_buffer(canvas1);
        usleep(delay);

        cl_queue.wait_and_throw();
    }
}
