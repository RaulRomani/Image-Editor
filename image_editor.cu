#include <cuda.h>
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <highgui.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cvui.h" // For interfaces

using namespace std;
using namespace cv;

# define PI 3.1416
# define WINDOW_NAME "ImageEditor"

#define CANVAS_WIDTH  960
#define CANVAS_HEIGHT 960

# define MENU_TRANSFORM         1
# define MENU_EQUALIZAR         2
# define MENU_CONVOLUCION       3
# define MENU_FOURIER                4
# define MENU_PATTERN_SEARCH    5
# define MENU_MORPH             6

int MENU        = 1;
int COLUMN_1    = 10;
int COLUMN_2    = 200;
int COLUMN_3    = 850;
int ROW_HEIGHT  = 35;
int ROW_ACTUAL  = 0;

dim3 blockDims;
dim3 gridDims;
cudaEvent_t start, stop;
unsigned char *gpu_image_output, *gpu_image_input, *gpu_image_temp, *gpu_image_aux;
unsigned char *cpu_image_output, *cpu_image_input, *cpu_image_aux;
Mat image_input, image_canvas, image_output, image_aux, image_temp;
bool pattern_loaded = false;
bool image_loaded = false;
float milliseconds = 0.0;
string file_path = "";

struct MatPixel {
    uchar b;
    uchar g;
    uchar r;
};

__global__ void kernel_Gaussiano(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    if (x < width - 3 && x > 3 && y < height - 3 && y > 3) {
        int p11 = offset - 2 * width - 2;
        int p12 = offset - 2 * width - 1;
        int p13 = offset - 2 * width;
        int p14 = offset - 2 * width + 1;
        int p15 = offset - 2 * width + 2;

        int p21 = offset - width - 2;
        int p22 = offset - width - 1;
        int p23 = offset - width;
        int p24 = offset - width + 1;
        int p25 = offset - width + 2;

        int p31 = offset - 2;
        int p32 = offset - 1;
        int p33 = offset;
        int p34 = offset + 1;
        int p35 = offset + 2;

        int p41 = offset + width - 2;
        int p42 = offset + width - 1;
        int p43 = offset + width;
        int p44 = offset + width + 1;
        int p45 = offset + width + 2;

        int p51 = offset + 2 * width - 2;
        int p52 = offset + 2 * width - 1;
        int p53 = offset + 2 * width;
        int p54 = offset + 2 * width + 1;
        int p55 = offset + 2 * width + 2;

        //if ( offset < height * width) {
        output_image[offset * 3] = (
                                       2 * input_image[p11 * 3] +  4 * input_image[p12 * 3]  +  5 * input_image[p13 * 3] +  4 * input_image[p14 * 3] + 2 * input_image[p15 * 3] +
                                       4 * input_image[p21 * 3] +  9 * input_image[p22 * 3]  + 12 * input_image[p23 * 3] +  9 * input_image[p24 * 3] + 4 * input_image[p25 * 3] +
                                       5 * input_image[p31 * 3] + 12 * input_image[p32 * 3]  + 15 * input_image[p33 * 3] + 12 * input_image[p34 * 3] + 5 * input_image[p35 * 3] +
                                       4 * input_image[p41 * 3] +  9 * input_image[p42 * 3]  + 12 * input_image[p43 * 3] +  9 * input_image[p44 * 3] + 4 * input_image[p45 * 3] +
                                       2 * input_image[p51 * 3] +  4 * input_image[p52 * 3]  +  5 * input_image[p53 * 3] +  4 * input_image[p54 * 3] + 2 * input_image[p55 * 3] ) / 159;
        output_image[offset * 3 + 1] = (
                                           2 * input_image[p11 * 3 + 1] +  4 * input_image[p12 * 3 + 1]  +  5 * input_image[p13 * 3 + 1] +  4 * input_image[p14 * 3 + 1] + 2 * input_image[p15 * 3 + 1] +
                                           4 * input_image[p21 * 3 + 1] +  9 * input_image[p22 * 3 + 1]  + 12 * input_image[p23 * 3 + 1] +  9 * input_image[p24 * 3 + 1] + 4 * input_image[p25 * 3 + 1] +
                                           5 * input_image[p31 * 3 + 1] + 12 * input_image[p32 * 3 + 1]  + 15 * input_image[p33 * 3 + 1] + 12 * input_image[p34 * 3 + 1] + 5 * input_image[p35 * 3 + 1] +
                                           4 * input_image[p41 * 3 + 1] +  9 * input_image[p42 * 3 + 1]  + 12 * input_image[p43 * 3 + 1] +  9 * input_image[p44 * 3 + 1] + 4 * input_image[p45 * 3 + 1] +
                                           2 * input_image[p51 * 3 + 1] +  4 * input_image[p52 * 3 + 1]  +  5 * input_image[p53 * 3 + 1] +  4 * input_image[p54 * 3 + 1] + 2 * input_image[p55 * 3 + 1] ) / 159;
        output_image[offset * 3 + 2] = (
                                           2 * input_image[p11 * 3 + 2] +  4 * input_image[p12 * 3 + 2]  +  5 * input_image[p13 * 3 + 2] +  4 * input_image[p14 * 3 + 2] + 2 * input_image[p15 * 3 + 2] +
                                           4 * input_image[p21 * 3 + 2] +  9 * input_image[p22 * 3 + 2]  + 12 * input_image[p23 * 3 + 2] +  9 * input_image[p24 * 3 + 2] + 4 * input_image[p25 * 3 + 2] +
                                           5 * input_image[p31 * 3 + 2] + 12 * input_image[p32 * 3 + 2]  + 15 * input_image[p33 * 3 + 2] + 12 * input_image[p34 * 3 + 2] + 5 * input_image[p35 * 3 + 2] +
                                           4 * input_image[p41 * 3 + 2] +  9 * input_image[p42 * 3 + 2]  + 12 * input_image[p43 * 3 + 2] +  9 * input_image[p44 * 3 + 2] + 4 * input_image[p45 * 3 + 2] +
                                           2 * input_image[p51 * 3 + 2] +  4 * input_image[p52 * 3 + 2]  +  5 * input_image[p53 * 3 + 2] +  4 * input_image[p54 * 3 + 2] + 2 * input_image[p55 * 3 + 2] ) / 159;

        //}
    } else {
        if (offset < width * height) {
            output_image[offset * 3]     = input_image[offset * 3];
            output_image[offset * 3 + 1] = input_image[offset * 3 + 1];
            output_image[offset * 3 + 2] = input_image[offset * 3 + 2];
        }
    }
}
__global__ void kernel_Media(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;

    if (x < width - 3 && x > 3 && y < height - 3 && y > 3) {
        int p11 = offset - 2 * width - 2;
        int p12 = offset - 2 * width - 1;
        int p13 = offset - 2 * width;
        int p14 = offset - 2 * width + 1;
        int p15 = offset - 2 * width + 2;

        int p21 = offset - width - 2;
        int p22 = offset - width - 1;
        int p23 = offset - width;
        int p24 = offset - width + 1;
        int p25 = offset - width + 2;

        int p31 = offset - 2;
        int p32 = offset - 1;
        int p33 = offset;
        int p34 = offset + 1;
        int p35 = offset + 2;

        int p41 = offset + width - 2;
        int p42 = offset + width - 1;
        int p43 = offset + width;
        int p44 = offset + width + 1;
        int p45 = offset + width + 2;

        int p51 = offset + 2 * width - 2;
        int p52 = offset + 2 * width - 1;
        int p53 = offset + 2 * width;
        int p54 = offset + 2 * width + 1;
        int p55 = offset + 2 * width + 2;

        output_image[offset * 3] = (
                                       1 * input_image[p11 * 3] + 1 * input_image[p12 * 3]  + 1 * input_image[p13 * 3] + 1 * input_image[p14 * 3] + 1 * input_image[p15 * 1] +
                                       1 * input_image[p21 * 3] + 1 * input_image[p22 * 3]  + 1 * input_image[p23 * 3] + 1 * input_image[p24 * 3] + 1 * input_image[p25 * 1] +
                                       1 * input_image[p31 * 3] + 1 * input_image[p32 * 3]  + 1 * input_image[p33 * 3] + 1 * input_image[p34 * 3] + 1 * input_image[p35 * 1] +
                                       1 * input_image[p41 * 3] + 1 * input_image[p42 * 3]  + 1 * input_image[p43 * 3] + 1 * input_image[p44 * 3] + 1 * input_image[p45 * 1] +
                                       1 * input_image[p51 * 3] + 1 * input_image[p52 * 3]  + 1 * input_image[p53 * 3] + 1 * input_image[p54 * 3] + 1 * input_image[p55 * 1] ) / 25;
        output_image[offset * 3 + 1] = (
                                           1 * input_image[p11 * 3 + 1] + 1 * input_image[p12 * 3 + 1]  + 1 * input_image[p13 * 3 + 1] + 1 * input_image[p14 * 3 + 1] + 1 * input_image[p15 * 3 + 1] +
                                           1 * input_image[p21 * 3 + 1] + 1 * input_image[p22 * 3 + 1]  + 1 * input_image[p23 * 3 + 1] + 1 * input_image[p24 * 3 + 1] + 1 * input_image[p25 * 3 + 1] +
                                           1 * input_image[p31 * 3 + 1] + 1 * input_image[p32 * 3 + 1]  + 1 * input_image[p33 * 3 + 1] + 1 * input_image[p34 * 3 + 1] + 1 * input_image[p35 * 3 + 1] +
                                           1 * input_image[p41 * 3 + 1] + 1 * input_image[p42 * 3 + 1]  + 1 * input_image[p43 * 3 + 1] + 1 * input_image[p44 * 3 + 1] + 1 * input_image[p45 * 3 + 1] +
                                           1 * input_image[p51 * 3 + 1] + 1 * input_image[p52 * 3 + 1]  + 1 * input_image[p53 * 3 + 1] + 1 * input_image[p54 * 3 + 1] + 1 * input_image[p55 * 3 + 1] ) / 25;
        output_image[offset * 3 + 2] = (
                                           1 * input_image[p11 * 3 + 2] + 1 * input_image[p12 * 3 + 2]  + 1 * input_image[p13 * 3 + 2] + 1 * input_image[p14 * 3 + 2] + 1 * input_image[p15 * 3 + 2] +
                                           1 * input_image[p21 * 3 + 2] + 1 * input_image[p22 * 3 + 2]  + 1 * input_image[p23 * 3 + 2] + 1 * input_image[p24 * 3 + 2] + 1 * input_image[p25 * 3 + 2] +
                                           1 * input_image[p31 * 3 + 2] + 1 * input_image[p32 * 3 + 2]  + 1 * input_image[p33 * 3 + 2] + 1 * input_image[p34 * 3 + 2] + 1 * input_image[p35 * 3 + 2] +
                                           1 * input_image[p41 * 3 + 2] + 1 * input_image[p42 * 3 + 2]  + 1 * input_image[p43 * 3 + 2] + 1 * input_image[p44 * 3 + 2] + 1 * input_image[p45 * 3 + 2] +
                                           1 * input_image[p51 * 3 + 2] + 1 * input_image[p52 * 3 + 2]  + 1 * input_image[p53 * 3 + 2] + 1 * input_image[p54 * 3 + 2] + 1 * input_image[p55 * 3 + 2] ) / 25;

    } else {
        if (offset < width * height) {
            output_image[offset * 3]     = input_image[offset * 3];
            output_image[offset * 3 + 1] = input_image[offset * 3 + 1];
            output_image[offset * 3 + 2] = input_image[offset * 3 + 2];
        }
    }
}
__device__ void sort(int* arr, int n) {
    int temp;
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
__global__ void kernel_Sobel(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    if (x < width - 1 && x > 2 && y < height - 1 && y > 2) {
        int p11 = offset - width - 1;
        int p12 = offset - width;
        int p13 = offset - width + 1;

        int p21 = offset - 1;
        int p22 = offset;
        int p23 = offset + 1;

        int p31 = offset + width - 1;
        int p32 = offset + width;
        int p33 = offset + width + 1;

        int auxInte1 = (    input_image[p11 * 3]  -   input_image[p13 * 3] +
                            2 * input_image[p21 * 3]  - 2 * input_image[p23 * 3] +
                            input_image[p31 * 3]  -   input_image[p33 * 3]) / 4;

        int auxInte2 = (   -input_image[p11 * 3] - 2 * input_image[p12 * 3] - input_image[p13 * 3] +
                           input_image[p31 * 3] + 2 * input_image[p32 * 3] + input_image[p33 * 3]) / 4;

        auxInte1 =  sqrtf((auxInte1 * auxInte1) + (auxInte2 * auxInte2))  ;

        if (auxInte1 > 255 ) {
            output_image[offset * 3]   = 255;
            output_image[offset * 3 + 1] = 255;
            output_image[offset * 3 + 2] = 255;
        } else if (auxInte1 < 0 ) {
            output_image[offset * 3]   = 0;
            output_image[offset * 3 + 1] = 0;
            output_image[offset * 3 + 2] = 0;
        }
        else {
            output_image[offset * 3]   = auxInte1;
            output_image[offset * 3 + 1] = auxInte1;
            output_image[offset * 3 + 2] = auxInte1;
        }
    } else {
        if (offset < width * height) {
            output_image[offset * 3]     = input_image[offset * 3];
            output_image[offset * 3 + 1] = input_image[offset * 3 + 1];
            output_image[offset * 3 + 2] = input_image[offset * 3 + 2];
        }
    }
}
__global__ void kernel_Mediana(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    if (x < width - 1 && x > 2 && y < height - 1 && y > 2) {


        int p11 = offset - width - 1;
        int p12 = offset - width;
        int p13 = offset - width + 1;

        int p21 = offset - 1;
        int p22 = offset;
        int p23 = offset + 1;

        int p31 = offset + width - 1;
        int p32 = offset + width;
        int p33 = offset + width + 1;

        int r[9];
        int g[9];
        int b[9];
        r[0] = input_image[p11 * 3];
        r[1] = input_image[p12 * 3];
        r[2] = input_image[p13 * 3];
        r[3] = input_image[p21 * 3];
        r[4] = input_image[p22 * 3];
        r[5] = input_image[p23 * 3];
        r[6] = input_image[p31 * 3];
        r[7] = input_image[p32 * 3];
        r[8] = input_image[p33 * 3];

        g[0] = input_image[p11 * 3 + 1];
        g[1] = input_image[p12 * 3 + 1];
        g[2] = input_image[p13 * 3 + 1];
        g[3] = input_image[p21 * 3 + 1];
        g[4] = input_image[p22 * 3 + 1];
        g[5] = input_image[p23 * 3 + 1];
        g[6] = input_image[p31 * 3 + 1];
        g[7] = input_image[p32 * 3 + 1];
        g[8] = input_image[p33 * 3 + 1];

        b[0] = input_image[p11 * 3 + 2];
        b[1] = input_image[p12 * 3 + 2];
        b[2] = input_image[p13 * 3 + 2];
        b[3] = input_image[p21 * 3 + 2];
        b[4] = input_image[p22 * 3 + 2];
        b[5] = input_image[p23 * 3 + 2];
        b[6] = input_image[p31 * 3 + 2];
        b[7] = input_image[p32 * 3 + 2];
        b[8] = input_image[p33 * 3 + 2];
        sort(r, 9);
        sort(g, 9);
        sort(b, 9);
        //if ( offset < height * width) {
        output_image[offset * 3] = r[5];
        output_image[offset * 3 + 1] = g[5];
        output_image[offset * 3 + 2] = b[5];
        //}
    } else {
        if (offset < width * height) {
            output_image[offset * 3]     = input_image[offset * 3];
            output_image[offset * 3 + 1] = input_image[offset * 3 + 1];
            output_image[offset * 3 + 2] = input_image[offset * 3 + 2];
        }
    }
}
__global__ void kernel_Equalizar(unsigned char *input_image, unsigned char *output_image, int width, int height, int r, int g, int b) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int new_r;
    int new_g;
    int new_b;
    if (offset < width * height) {
        new_r = input_image[offset * 3] + r;
        new_g = input_image[offset * 3 + 1] + g;
        new_b = input_image[offset * 3 + 2] + b;
        if (new_r < 0 ) {
            output_image[offset * 3] = 0;
        } else if (new_r > 255 ) {
            output_image[offset * 3] = 255;
        } else {
            output_image[offset * 3] = new_r;
        }
        if (new_g < 0 ) {
            output_image[offset * 3 + 1] = 0;
        } else if (new_g > 255 ) {
            output_image[offset * 3 + 1] = 255;
        } else {
            output_image[offset * 3 + 1] = new_g;
        }
        if (new_b < 0 ) {
            output_image[offset * 3 + 2] = 0;
        } else if (new_b > 255 ) {
            output_image[offset * 3 + 2] = 255;
        } else {
            output_image[offset * 3 + 2] = new_b;
        }
    }
}
__global__ void kernel_Equalizar_2(unsigned char *input_image, unsigned char *output_image, int width, int height, int r, float r_factor, int g, float g_factor, int b, float b_factor) {
    int temp = width;
    width = height;
    height = temp;
    int new_r;
    int new_g;
    int new_b;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < width * height) {
        new_r = (input_image[offset * 3] - r) * r_factor;
        new_g = (input_image[offset * 3 + 1] - g) * g_factor;
        new_b = (input_image[offset * 3 + 2] - b) * b_factor;
        if (new_r < 0 ) {
            output_image[offset * 3] = 0;
        } else if (new_r > 255 ) {
            output_image[offset * 3] = 255;
        } else {
            output_image[offset * 3] = new_r;
        }
        if (new_g < 0 ) {
            output_image[offset * 3 + 1] = 0;
        } else if (new_g > 255 ) {
            output_image[offset * 3 + 1] = 255;
        } else {
            output_image[offset * 3 + 1] = new_g;
        }
        if (new_b < 0 ) {
            output_image[offset * 3 + 2] = 0;
        } else if (new_b > 255 ) {
            output_image[offset * 3 + 2] = 255;
        } else {
            output_image[offset * 3 + 2] = new_b;
        }
    }
}
__global__ void kernel_rotate(unsigned char * input_image, unsigned char * output_image, int width, int height, float sn, float cs) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    int xc = width / 2;
    int yc = height / 2;
    int newx = ((float)x - xc) * cs - ((float)y - yc) * sn + xc;
    int newy = ((float)x - xc) * sn + ((float)y - yc) * cs + yc;
    int new_offset = newx  + newy * width;
    if(x < width && y<height && newx < width && newx > 0 && newy < height && newy > 0){
        output_image[offset * 3 + 0] = input_image[new_offset * 3 + 0];
        output_image[offset * 3 + 1] = input_image[new_offset * 3 + 1];
        output_image[offset * 3 + 2] = input_image[new_offset * 3 + 2];
    }
}
__global__ void kernel_resize(unsigned char *input_image, unsigned char *output_image, int width, int height, int width_out, int height_out) {
    int temp = width;
    width = height;
    height = temp;
    temp = width_out;
    width_out = height_out;
    height_out = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width_out;
    int y = (offset - x) / width_out;
    float scale = width_out * 1.0 / width;
    int newx = x / scale;
    int newy = y / scale;
    int new_offset;

    if ( offset < height_out * width_out) {
        new_offset = newx  + newy * width;
        output_image[offset * 3 + 0] = input_image[new_offset * 3 + 0];
        output_image[offset * 3 + 1] = input_image[new_offset * 3 + 1];
        output_image[offset * 3 + 2] = input_image[new_offset * 3 + 2];
    }
}
__global__ void kernel_pattern_search(unsigned char *input_image, unsigned char *output_image, unsigned char * pattern_image, int width, int height, int width_pattern, int height_pattern,double percent) {
    int temp = width;
    width = height;
    height = temp;
    temp = width_pattern;
    width_pattern = height_pattern;
    height_pattern = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int off_input = offset;
    int off_pattern = 0;
    int hit_count = 0;
    if (offset < width * height - width_pattern * height_pattern) {
        for (int j = 0; j < height_pattern; j++) {
            for ( int i = 0; i < width_pattern; i++) {
                if ( input_image[off_input * 3]     == pattern_image[off_pattern * 3 ] &&
                        input_image[off_input * 3 + 1] == pattern_image[off_pattern * 3 + 1] &&
                        input_image[off_input * 3 + 2] == pattern_image[off_pattern * 3 + 2] ) {
                    hit_count++;
                }
                off_input++;
                off_pattern++;
            }
            off_input += width - width_pattern;
        }
        off_input = offset;
        if (hit_count >= width_pattern * height_pattern * percent) {
            for (int j = 0; j < height_pattern; j++) {
                for ( int i = 0; i < width_pattern; i++) {
                    output_image[off_input * 3]     =   0 + 0.5*input_image[off_input * 3]; //pattern_image[off_pattern * 3 ] ;
                    output_image[off_input * 3 + 1] = 128 + 0.5*input_image[off_input * 3 + 1]; //pattern_image[off_pattern * 3 + 1] ;
                    output_image[off_input * 3 + 2] =   0 + 0.5*input_image[off_input * 3 + 2]; //pattern_image[off_pattern * 3 + 2] ;
                    off_input++;
                }
                off_input += width - width_pattern;
            }
        }
    }
}
__global__ void kernel_image_to_gray(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int new_c;
    if (offset < width * height) {
        new_c = (   input_image[offset * 3] +
                    input_image[offset * 3 + 1] +
                    input_image[offset * 3 + 2] ) / 3;
        output_image[offset * 3]     = new_c;
        output_image[offset * 3 + 1] = new_c;
        output_image[offset * 3 + 2] = new_c;
    }
}
__global__ void kernel_image_erosion(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    int new_c = 255;
    if (x < width - 1 && x > 2 && y < height - 1 && y > 2) {
        int p12 = offset - width;
        int p21 = offset - 1;
        int p22 = offset;
        int p23 = offset + 1;
        int p32 = offset + width;

        int c12 = ( input_image[p12 * 3] +
                    input_image[p12 * 3 + 1] +
                    input_image[p12 * 3 + 2] ) / 3 / 128;
        int c21 = ( input_image[p21 * 3] +
                    input_image[p21 * 3 + 1] +
                    input_image[p21 * 3 + 2] ) / 3 / 128;
        int c22 = ( input_image[p22 * 3] +
                    input_image[p22 * 3 + 1] +
                    input_image[p22 * 3 + 2] ) / 3 / 128;
        int c23 = ( input_image[p23 * 3] +
                    input_image[p23 * 3 + 1] +
                    input_image[p23 * 3 + 2] ) / 3 / 128;
        int c32 = ( input_image[p32 * 3] +
                    input_image[p32 * 3 + 1] +
                    input_image[p32 * 3 + 2] ) / 3 / 128;
        int cant_hits = 5 - (c12 + c21 + c22 + c23 + c32);

        if (cant_hits == 5) {
            new_c = 0;
        }
        int umbral = 128;
        if ( offset < height * width) {
            output_image[offset * 3]     = new_c;
            output_image[offset * 3 + 1] = new_c;
            output_image[offset * 3 + 2] = new_c;
        }
    } else {
        if (offset < width * height) {
            new_c = (   input_image[offset * 3] +
                        input_image[offset * 3 + 1] +
                        input_image[offset * 3 + 2] ) / 3 / 128 * 255;
            output_image[offset * 3]     = new_c;
            output_image[offset * 3 + 1] = new_c;
            output_image[offset * 3 + 2] = new_c;
        }
    }
}
__global__ void kernel_image_dilatation(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;
    int new_c = 255;

    if (x < width - 1 && x > 2 && y < height - 1 && y > 2) {
        int p12 = offset - width;
        int p21 = offset - 1;
        int p22 = offset;
        int p23 = offset + 1;
        int p32 = offset + width;

        int c12 = ( input_image[p12 * 3] +
                    input_image[p12 * 3 + 1] +
                    input_image[p12 * 3 + 2] ) / 3 / 128;
        int c21 = ( input_image[p21 * 3] +
                    input_image[p21 * 3 + 1] +
                    input_image[p21 * 3 + 2] ) / 3 / 128;
        int c22 = ( input_image[p22 * 3] +
                    input_image[p22 * 3 + 1] +
                    input_image[p22 * 3 + 2] ) / 3 / 128;
        int c23 = ( input_image[p23 * 3] +
                    input_image[p23 * 3 + 1] +
                    input_image[p23 * 3 + 2] ) / 3 / 128;
        int c32 = ( input_image[p32 * 3] +
                    input_image[p32 * 3 + 1] +
                    input_image[p32 * 3 + 2] ) / 3 / 128;
        int cant_hits = 5 - ( c12 + c21 + c22 + c23 + c32);
        if (cant_hits > 1) {
            new_c = 0;
        }
        int umbral = 128;
        if ( offset < height * width) {
            output_image[offset * 3]     = new_c;
            output_image[offset * 3 + 1] = new_c;
            output_image[offset * 3 + 2] = new_c;
        }
    } else {
        if (offset < width * height) {
            new_c = (   input_image[offset * 3] +
                        input_image[offset * 3 + 1] +
                        input_image[offset * 3 + 2] ) / 3 / 128 * 255;
            output_image[offset * 3]     = new_c;
            output_image[offset * 3 + 1] = new_c;
            output_image[offset * 3 + 2] = new_c;
        }
    }
}
__global__ void kernel_image_to_bn(unsigned char *input_image, unsigned char *output_image, int width, int height) {
    int temp = width;
    width = height;
    height = temp;
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int new_c;
    if (offset < width * height) {
        new_c = (   input_image[offset * 3] +
                    input_image[offset * 3 + 1] +
                    input_image[offset * 3 + 2] ) / 3 / 128 * 255;
        /*if (new_c > 128) {
            new_c = 255;
        } else {
            new_c = 0;
        }*/
        output_image[offset * 3]     = new_c;
        output_image[offset * 3 + 1] = new_c;
        output_image[offset * 3 + 2] = new_c;
    }
}
void loadArrayFromMat(Mat& imagen, unsigned char* array, int width, int height) {
    int i = 0;
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            MatPixel& pixel = imagen.at<MatPixel>(h, w);
            array[i++] = pixel.r;
            array[i++] = pixel.g;
            array[i++] = pixel.b;
        }
    }
}
void loadMatFromArray(Mat& imagen, unsigned char* array, int width, int height) {
    int i = 0;
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            MatPixel& pixel = imagen.at<MatPixel>(h, w);
            pixel.r = array[i++];
            pixel.g = array[i++];
            pixel.b = array[i++];
        }
    }
}
vector<complex<double>> fft(vector<complex<double>>& a) {
    int n = a.size();

    if (n == 1)
        return vector<complex<double>>(1, a[0]);

    vector<complex<double>> w(n);
    for (int i = 0; i < n; i++) {
        double alpha = 2 * M_PI * i / n;
        w[i] = complex<double>(cos(alpha), sin(alpha));
    }

    vector<complex<double>> even_elements(n / 2), odd_elements(n / 2);
    for (int i = 0; i < n / 2; i++) {
        even_elements[i] = a[i * 2];
        odd_elements[i] = a[i * 2 + 1];
    }

    vector<complex<double>> y0 = fft(even_elements);
    vector<complex<double>> y1 = fft(odd_elements);

    vector<complex<double>> y(n);

    for (int k = 0; k < n / 2; k++) {
        y[k] = y0[k] + w[k] * y1[k];
        y[k + n / 2] = y0[k] - w[k] * y1[k];
    }
    return y;
}
void EqualizarHistograma(unsigned char *cpu_input_image, unsigned char *input_image, unsigned char* output_image, int width, int height) {
    //int temp = width;
    //width = height;
    //height = temp;
    int *r = new int[256];
    int *g = new int[256];
    int *b = new int[256];
    for (int i = 0; i < 256; i++) {
        r[i] = 0;
        g[i] = 0;
        b[i] = 0;
    }
    int new_r, new_g, new_b;
    for (int i = 0; i < width * height; i++) {
        new_r = cpu_input_image[i * 3];
        new_g = cpu_input_image[i * 3 + 1] ;
        new_b = cpu_input_image[i * 3 + 2] ;
        if (new_r < 0 ) {
            r[0]++;
        } else if (new_r > 255 ) {
            r[255]++;
        } else {
            r[new_r]++;
        }
        if (new_g < 0 ) {
            b[0]++;
        } else if (new_g > 255 ) {
            b[255]++;
        } else {
            b[new_g]++;
        }
        if (new_b < 0 ) {
            b[0]++;
        } else if (new_b > 255 ) {
            b[255]++;
        } else {
            b[new_b]++;
        }
    }
    int r_max = 0, r_min = 255;
    int g_max = 0, g_min = 255;
    int b_max = 0, b_min = 255;
    for (int i = 0; i < 10; i++) {
        if (r[i] > 0) {
            if (i < r_min)r_min = i;
            if (i > r_max)r_max = i;
        }
        if (g[i] > 0) {
            if (i < g_min)g_min = i;
            if (i > g_max)g_max = i;
        }
        if (b[i] > 0) {
            if (i < b_min)b_min = i;
            if (i > b_max)b_max = i;
        }
    }
    float r_factor = 256;
    if (r_max - r_min > 0)
        r_factor /= (r_max - r_min);
    else r_factor = 1;
    float g_factor = 256;
    if (g_max - g_min > 0)
        g_factor /= (g_max - g_min);
    else g_factor = 1;
    float b_factor = 256;
    if (b_max - b_min > 0)
        b_factor /= (b_max - b_min);
    else b_factor = 1;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    cout << "running kernel" << endl;
    kernel_Equalizar_2 <<< gridDims, blockDims>>>(input_image, output_image, width, height, r_min, r_factor, g_min, g_factor, b_min, b_factor);
    cout << r_min << " " << g_min << " " << b_min << endl;
    //kernel_Equalizar<<< gridDims, blockDims>>>(input_image, output_image, width, height,1,1,1);
    cout << "Finish kernel " << endl;
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
}
void applyKernelGaussiano(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_Gaussiano <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
    //kernel_Gaussiano<<<gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height);
    //kernel_Equalizar <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, r, g, b);
    //kernel_rotate <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, sn, cs);
}
void applyKernelMedia(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_Media <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
}
void applyKernelMediana(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_Mediana <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
}
void applyKernelGray(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_image_to_gray <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
}
void applyKernelBinary(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    //cudaMemcpy( gpu_image_output, cpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    //cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_image_to_bn <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
}
void applyKernelEqualizar(unsigned char* input_image, unsigned char* output_image, int width, int height, int r, int g, int b) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_Equalizar <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, r, g, b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp   = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input  = gpu_image_temp;
}
void applyKernelErosion(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_image_erosion <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp   = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input  = gpu_image_temp;
}
void applyKernelSobel(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_Sobel <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp   = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input  = gpu_image_temp;
}
void action_rotate(unsigned char* input_image, unsigned char* output_image, int width, int height,float sn,float cs){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_rotate <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, sn, cs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    /*gpu_image_temp   = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input  = gpu_image_temp;*/
}
void applyKernelDilatation(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    //cudaMemcpy( gpu_image_output, cpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    //cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_image_dilatation <<< gridDims, blockDims>>>(input_image, output_image, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    gpu_image_temp = gpu_image_output;
    gpu_image_output = gpu_image_input;
    gpu_image_input = gpu_image_temp;
}
void action_search_pattern(/*unsigned char* input_image, unsigned char* output_image,unsigned char* patter_image,*/ int width, int height, int width_pattern, int height_pattern,double percent) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_pattern_search <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, gpu_image_aux, width, height, width_pattern, height_pattern,percent/100.0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << endl;
    //gpu_image_temp = gpu_image_output;
    //gpu_image_output = gpu_image_input;
    //gpu_image_input = gpu_image_temp;
    cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
    loadMatFromArray(image_output, cpu_image_output, width, height);
    //kernel_Gaussiano<<<gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height);
    //kernel_Equalizar <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, r, g, b);
    //kernel_rotate <<< gridDims, blockDims>>>(gpu_image_input, gpu_image_output, width, height, sn, cs);
}
void generateNoise(unsigned char* input_image, int width, int height, float percent) {
    int noise_pixels = width * height * percent / 100;
    int n_w;
    int n_h;
    int offset;
    for (int i = 0; i < noise_pixels; i++) {
        n_w = rand() % width;
        n_h = rand() % height;
        offset = n_w * width + n_h;
        if (offset < width * height) {
            input_image[offset * 3] = 0;
            input_image[offset * 3 + 1] = 0;
            input_image[offset * 3 + 2] = 0;
        }
    }
    cudaMemcpy( cpu_image_input, gpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
}
Mat calcHist(Mat& image) {
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    vector<Mat> bgr_planes;
    split( image, bgr_planes );
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    int hist_w = 180; int hist_h = 150;
    int bin_w = cvRound( (double) hist_w / histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0, 0, 0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for ( int i = 1; i < histSize; i++ ) {
        line( histImage, Point( bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1)) ) ,
              Point( bin_w * (i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1)) ) ,
              Point( bin_w * (i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1)) ) ,
              Point( bin_w * (i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    return histImage;
}
Mat calcFastFourierTransform(Mat& image, int width, int height) {
    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );
    //int i = 0;
    vector<complex<double>> cols(width);
    vector<complex<double>> rows(height);
    vector<complex<double>> temp;
    vector<vector<complex<double>>> fftMat;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            cols[w] = {(double)gray_image.at<uchar>(h, w), 0.0};
        }
        temp = fft(cols);
        fftMat.push_back(temp);
    }

    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            cols[h] = fftMat[h][w];
        }
        temp = fft(cols);
        for (int h = 0; h < height; h++) {
            gray_image.at<uchar>(h, w) = log(abs(temp[h]) + 1);
        }
    }

    gray_image = gray_image(Rect(0, 0, gray_image.cols & -2, gray_image.rows & -2));

    int cx = gray_image.cols / 2;
    int cy = gray_image.rows / 2;

    Mat q0(gray_image, Rect(0, 0, cx, cy));
    Mat q1(gray_image, Rect(cx, 0, cx, cy));
    Mat q2(gray_image, Rect(0, cy, cx, cy));
    Mat q3(gray_image, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(gray_image, gray_image, 0, 255, CV_MINMAX);

    return gray_image;
}
void actionShowFFT(Mat& image, int width, int height) {
    Mat image_FFT =  calcFastFourierTransform(image, width, height);
    imshow("Fast Fourier Transform", image_FFT);
}
void action_open_file(Mat &image) {
    FILE *in;
    if (!(in = popen("zenity  --title=\"Seleccionar imagen\" --file-selection", "r"))) {
        image_loaded = false;
        return;
    }
    image_loaded = true;
    char buff[512];
    file_path = "";
    while (fgets(buff, sizeof(buff), in) != NULL) {
        file_path += buff;
    }
    pclose(in);

    //remove the "\n"
    file_path.erase(std::remove(file_path.begin(), file_path.end(), '\n'), file_path.end());

    // path + filename + format
    image = imread(file_path);
    int width = image.cols;
    int height = image.rows;

    delete cpu_image_input;
    delete cpu_image_output;
    cpu_image_output = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    cpu_image_input = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    cudaFree(gpu_image_output);
    cudaFree(gpu_image_input);
    cudaMalloc( (void**)&gpu_image_output, width * height * 3 * sizeof(unsigned char));
    cudaMalloc( (void**)&gpu_image_input, width * height * 3 * sizeof(unsigned char) );

    loadArrayFromMat(image, cpu_image_input, width, height);
    ////loadMatFromArray()
    //
    //image_output = Mat(height, width, CV_8UC3, Scalar(255, 0, 0));
    //image_output = image_input;
    //cudaMemcpy( gpu_image_output, cpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_image_output, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    blockDims = dim3(512, 1, 1);
    gridDims = dim3((unsigned int) ceil((double)(width * height * 3 / blockDims.x)), 1, 1 );
    //int rotate_angle = 0;
}
void action_open_file_aux(Mat &image) {
    FILE *in;
    if (!(in = popen("zenity  --title=\"Seleccionar imagen\" --file-selection", "r"))) {
        return;
    }

    char buff[512];
    string selectFile = "";
    while (fgets(buff, sizeof(buff), in) != NULL) {
        selectFile += buff;
    }
    pclose(in);

    //remove the "\n"
    selectFile.erase(std::remove(selectFile.begin(), selectFile.end(), '\n'), selectFile.end());

    // path + filename + format
    image = imread(selectFile);
    int width = image.cols;
    int height = image.rows;

    delete cpu_image_aux;
    //delete cpu_image_output;
    cpu_image_aux = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    //cpu_image_input = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    cudaFree(gpu_image_aux);
    //cudaFree(gpu_image_input);
    cudaMalloc( (void**)&gpu_image_aux, width * height * 3 * sizeof(unsigned char));
    //cudaMalloc( (void**)&gpu_image_input, width * height * 3 * sizeof(unsigned char) );

    loadArrayFromMat(image, cpu_image_aux, width, height);
    ////loadMatFromArray()
    //
    //image_output = Mat(height, width, CV_8UC3, Scalar(255, 0, 0));
    //image_output = image_input;
    cudaMemcpy( gpu_image_aux, cpu_image_aux, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    //cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    //blockDims = dim3(512, 1, 1);
    //gridDims = dim3((unsigned int) ceil((double)(width * height * 3 / blockDims.x)), 1, 1 );
    //int rotate_angle = 0;
}
void reload_image(Mat &image) {
    // path + filename + format
    image = imread(file_path);
    int width = image.cols;
    int height = image.rows;

    delete cpu_image_input;
    delete cpu_image_output;
    cpu_image_output = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    cpu_image_input = (unsigned char*)malloc(sizeof(unsigned char) * height * width * 3);
    cudaFree(gpu_image_output);
    cudaFree(gpu_image_input);
    cudaMalloc( (void**)&gpu_image_output, width * height * 3 * sizeof(unsigned char));
    cudaMalloc( (void**)&gpu_image_input, width * height * 3 * sizeof(unsigned char) );

    loadArrayFromMat(image, cpu_image_input, width, height);
    ////loadMatFromArray()
    //
    //image_output = Mat(height, width, CV_8UC3, Scalar(255, 0, 0));
    //image_output = image_input;
    //cudaMemcpy( gpu_image_output, cpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_image_output, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
    blockDims = dim3(512, 1, 1);
    gridDims = dim3((unsigned int) ceil((double)(width * height * 3 / blockDims.x)), 1, 1 );
    //int rotate_angle = 0;
}
int main( int argc, char** argv ) {
    char * message = "";
    unsigned int theColor = 0xff0000;
    Mat frame = cv::Mat(980, 1170, CV_8UC3);
    Mat image_histogram, image_FFT, image_scale;
    //Mat image_temp = cv::Mat(CANVAS_WIDTH, CANVAS_HEIGHT, CV_8UC3, Scalar(40, 40, 40));

    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);
    char tecla;
    double r = 0, g = 0, b = 0;
    double r_l = 0, g_l = 0, b_l = 0;
    double percent = 80;
    int height  = CANVAS_HEIGHT;
    int width   = CANVAS_WIDTH;
    //int rotate ;
    int rotate_angle = 0;
    int rotate_angle_l = 0;
    double scale = 1;
    double scale_l = 1;
    float sn,cs,deg;

    //int degree = 0;
    //float zoom = 1;
    image_canvas = cv::Mat(CANVAS_WIDTH, CANVAS_HEIGHT, CV_8UC3, Scalar(40, 40, 40));

    image_output = cv::Mat(CANVAS_WIDTH, CANVAS_HEIGHT, CV_8UC3);
    image_output = cv::Scalar(40, 40, 40);

    while (1) {
        ROW_ACTUAL = 0;
        frame = cv::Scalar(49, 52, 49);
        if (cpu_image_input) {
            //applyKernelEqualizar(gpu_image_input, gpu_image_output, width, height, r, g, b);
        }
        cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 2 - 5, "CARGAR IMAGEN");
        if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Abrir imagen")) {
            action_open_file(image_output);
            width = image_output.cols;
            height = image_output.rows;
            r = 0, g = 0, b = 0;
            r_l = 0, g_l = 0, b_l = 0;
            rotate_angle = 0,rotate_angle_l=0;
            scale = 1, scale_l = 1;
        }
        if (image_loaded) {
            if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Volver a cargar imagen")) {
                reload_image(image_output);
                width = image_output.cols;
                height = image_output.rows;
                r = 0, g = 0, b = 0;
                r_l = 0, g_l = 0, b_l = 0;
                rotate_angle = 0,rotate_angle_l=0;
                scale = 1, scale_l = 1;
            }
            if (MENU == MENU_TRANSFORM) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 2 - 5, "TRANSFORMACIONES");
                cvui::counter(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, &rotate_angle,5);
                cvui::counter(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, &scale,0.5);
                
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Convertir a grises")) {
                    applyKernelGray(gpu_image_input, gpu_image_output, width, height);
                    r = 0, g = 0, b = 0;
                    r_l = 0, g_l = 0, b_l = 0;
                    //rotate_angle = 0,rotate_angle_l=0;
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Convertir a b/n")) {
                    applyKernelBinary(gpu_image_input, gpu_image_output, width, height);
                    //r = 0, g = 0, b = 0;
                    r = 0, g = 0, b = 0;
                    r_l = 0, g_l = 0, b_l = 0;
                    //rotate_angle = 0,rotate_angle_l=0;
                }
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "TRANSFORMACIONES");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_TRANSFORM;
                }
            }
            if (MENU == MENU_EQUALIZAR) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 5 - 5, "EQUALIZAR");
                cvui::trackbar(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, 150, &r, -250.0, 250.0);
                cvui::trackbar(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, 150, &g, -250.0, 250.0);
                cvui::trackbar(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, 150, &b, -250.0, 250.0);
                /*if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Equalizar")) {
                    EqualizarHistograma(cpu_image_input, gpu_image_input, gpu_image_output, width, height);
                }*/
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "EQUALIZAR");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_EQUALIZAR;
                }
            }
            if (MENU == MENU_CONVOLUCION) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 5 - 5, "CONVOLUCION");
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Sobel")) {
                    applyKernelSobel(gpu_image_input, gpu_image_output, width, height);
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Suavizado Gaussiano")) {
                    applyKernelGaussiano(gpu_image_input, gpu_image_output, width, height);
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Suavizado Media")) {
                    applyKernelMedia(gpu_image_input, gpu_image_output, width, height);
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Suavizado Mediana")) {
                    applyKernelMediana(gpu_image_input, gpu_image_output, width, height);
                    //imshow("result",image_output);
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Agregar ruido")) {
                    generateNoise(cpu_image_input, width, height, 10);
                    loadMatFromArray(image_output, cpu_image_input, width, height);
                    cudaMemcpy( gpu_image_input, cpu_image_input, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
                    //imageOutput = Mat(height, width, CV_8UC3, Scalar(255, 0, 0));
                    //imageOutput = imageInput;
                    //cudaMemcpy( gpu_image_output, cpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice );
                    //cudaMemcpy( cpu_image_output, gpu_image_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
                }
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "CONVOLUCION");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_CONVOLUCION;
                }
            }           
            if (MENU == MENU_FOURIER) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 5 - 5, "FOURIER");
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Fast Fourier Transform")) {
                    actionShowFFT(image_output, width, height);
                }
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "FOURIER");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_FOURIER;
                }
            }
            if (MENU == MENU_MORPH) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 5 - 5, "MORFOLOGIA");
                
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Erosion")) {
                    applyKernelErosion(gpu_image_input, gpu_image_output, width, height);
                    r = 0, g = 0, b = 0;
                    r_l = 0, g_l = 0, b_l = 0;
                    //rotate_angle = 0,rotate_angle_l=0;
                }
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Dilatacion")) {
                    applyKernelDilatation(gpu_image_input, gpu_image_output, width, height);
                    r = 0, g = 0, b = 0;
                    r_l = 0, g_l = 0, b_l = 0;
                    //rotate_angle = 0,rotate_angle_l=0;
                }
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "MORFOLOGIA");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_MORPH;
                }
            }
            if (MENU == MENU_PATTERN_SEARCH) {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL++, COLUMN_2 - 10, ROW_HEIGHT * 5 - 5, "BUSCAR PATRON");
                if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Cargar patron")) {
                    action_open_file_aux(image_aux);
                    pattern_loaded = true;
                }
                if (pattern_loaded) {
                    int aux_width  = image_aux.cols;
                    int aux_height = image_aux.rows;
                    //imshow("Pattern",image_aux);
                    float factor;
                    int desp_x, desp_y;
                    int PATTER_WIDTH  = 180;
                    int PATTER_HEIGHT = 180;
                    if (aux_width < aux_height) {
                        factor = PATTER_HEIGHT * 1.0 / aux_height;
                        desp_x = (PATTER_WIDTH - aux_width * factor ) / 2;
                        desp_y = 0;
                    } else {
                        factor = PATTER_WIDTH  * 1.0 / aux_width;
                        desp_x = 0;
                        desp_y = (PATTER_HEIGHT - aux_height * factor) / 2;
                    }
                    cv::resize(image_aux, image_temp, Size(aux_width * factor, aux_height * factor)); //, interpolation = cv.INTER_CUBIC);
                    //image_temp.copyTo(image_canvas(cv::Rect(desp_x, desp_y, image_temp.cols, image_temp.rows)));
                    cvui::image(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, image_temp);
                    ROW_ACTUAL += 4;
                    cvui::trackbar(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, 150, &percent, 0.0, 100.0);
                    if (cvui::button(frame, COLUMN_1, ROW_HEIGHT * ROW_ACTUAL++, "Buscar en la imagen")) {
                        action_search_pattern(width, height, aux_width, aux_height,percent);
                    }
                }
            } else {
                cvui::window(frame, COLUMN_1 / 2, 5 + ROW_HEIGHT * ROW_ACTUAL, COLUMN_2 - 10, ROW_HEIGHT - 5, "BUSCAR PATRON");
                if (cvui::button(frame, COLUMN_2 - 25, 5 + ROW_HEIGHT * ROW_ACTUAL++, 20, 20, "+")) {
                    MENU = MENU_PATTERN_SEARCH;
                }
            }

            if(rotate_angle !=0){
                cudaMemset( gpu_image_output, 0, width * height * 3 * sizeof(unsigned char) );
                deg = rotate_angle * PI / 180;
                sn = sin(deg);
                cs = cos(deg);
                action_rotate(gpu_image_input, gpu_image_output, width, height,sn,cs);
            }
            /*if (r != 0 || g != 0 || b != 0) {
                applyKernelEqualizar(gpu_image_input, gpu_image_output, width, height, r_l - r, g_l - g, b_l - b);
            }*/
            /*if(rotate_angle - rotate_angle_l !=0){
                cudaMemset( gpu_image_output, 0, width * height * 3 * sizeof(unsigned char) );
                deg = (rotate_angle -rotate_angle_l ) * PI / 180;
                sn = sin(deg);
                cs = cos(deg);
                action_rotate(gpu_image_input, gpu_image_output, width, height,sn,cs);
                rotate_angle_l = rotate_angle;
            }*/
            if (r_l - r != 0 || g_l - g != 0 || b_l - b != 0) {
                applyKernelEqualizar(gpu_image_input, gpu_image_output, width, height, r_l - r, g_l - g, b_l - b);
                r_l = r;
                g_l = g;
                b_l = b;
            }
            if(scale_l - scale != 0){
                int width_zoom = width * scale;
                int height_zoom = height * scale;
                image_scale = Mat(height_zoom, width_zoom, CV_8UC3, Scalar(0, 0, 0));
                unsigned char *gpu_imgScale;
                unsigned char *cpu_imgScale = (unsigned char*)malloc(sizeof(unsigned char) * height_zoom * width_zoom * 3);
                cudaMalloc( (void**)&gpu_imgScale, width_zoom * height_zoom * 3 * sizeof(unsigned char));
                
                dim3 blockDimsScale(512, 1, 1);
                dim3 gridDimsScale((unsigned int) ceil((double)(width_zoom * height_zoom * 3 / blockDims.x)), 1, 1 );
                kernel_resize <<< gridDimsScale, blockDimsScale>>>(gpu_image_input, gpu_imgScale, width, height, width_zoom, height_zoom);
                cudaMemcpy( cpu_imgScale, gpu_imgScale, width_zoom * height_zoom * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
                loadMatFromArray(image_scale, cpu_imgScale, width_zoom, height_zoom);
                cudaFree( gpu_imgScale);
                imshow( "Scale", image_scale );
                scale_l = scale;
            }
        }
        image_canvas = cv::Mat(CANVAS_WIDTH, CANVAS_HEIGHT, CV_8UC3, Scalar(40, 40, 40));
        float factor;
        int desp_x, desp_y;
        if (width < height) {
            factor = CANVAS_HEIGHT * 1.0 / height;
            desp_x = (CANVAS_WIDTH - width * factor ) / 2;
            desp_y = 0;
        } else {
            factor = CANVAS_WIDTH  * 1.0 / width;
            desp_x = 0;
            desp_y = (CANVAS_HEIGHT - height * factor) / 2;
        }
        cv::resize(image_output, image_temp, Size(width * factor, height * factor)); //, interpolation = cv.INTER_CUBIC);
        image_temp.copyTo(image_canvas(cv::Rect(desp_x, desp_y, image_temp.cols, image_temp.rows)));
        cvui::image(frame, COLUMN_2, 10, image_canvas);
        image_histogram =  calcHist(image_output);

        //cvui::text(frame, COLUMN_1, 780, message,0.5,theColor);
        cvui::printf(frame, COLUMN_1, 780, "Finished in(ms)  = %.2f", milliseconds);
        cvui::window(frame, COLUMN_1 / 2, 800, COLUMN_2 - 10, 170, "Histograma");
        cvui::image(frame, COLUMN_1, 820, image_histogram);

        cvui::update();
        cv::imshow(WINDOW_NAME, frame);
        tecla = waitKey(50);
        if (tecla == 27) {
            break;
        }
    }
    return 0;
}