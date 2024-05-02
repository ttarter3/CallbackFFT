//
// Created by ttarter on 3/29/24.
//

#ifndef CUDATESTSTRING_DEFINE_H
#define CUDATESTSTRING_DEFINE_H

#define SIZE2 2
#define SIZE4 4
#define SIZE8 8

#define TPB_2048 2048
#define TPB_1024 1024
#define TPB_512 512
#define TPB_256 256
#define TPB_128 128
#define TPB_64 64
#define WARP 32
#define MAX_COALESCENCE TPB_512
#define SMALL_SET TPB_64
#define BIG_SET TPB_1024

#define TILE_SIZE TPB_512

#define BATCH 1

#endif //CUDATESTSTRING_DEFINE_H
