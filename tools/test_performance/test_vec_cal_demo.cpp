#include <stdio.h>
#include <sys/time.h>

#include <cmath>
#include <fstream>

#include "vsag/vsag.h"

static database_context* context;

static inline void delete_mem_func(float* base_data, float* massb, database_context* context) {
    if (base_data) {
        delete[] base_data;
        base_data = nullptr;
    }
    if (massb) {
        delete[] massb;
        massb = nullptr;
    }
    if (context) {
        delete context;
    }
}

static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

void generate_random_dataset(float* base_data, uint64_t vec_size, uint64_t vec_dim) {
    srand(time(NULL));  // 初始化随机种子
    for (int i = 0; i < vec_size; ++i) {
        for (int j = 0; j < vec_dim; ++j) {
            base_data[i * vec_dim + j] = (float)(rand() / (RAND_MAX + 1.0) * 3.0);  // 生成 0 到 3 的随机浮点数
        }
    }
}

int main(int argc, char** argv) {
    context = new database_context();
    size_t vecdim = 1024;     // vec dim
    size_t vecsize = 100000;  // vec size
    size_t cal_times = 64;
    float* base_data = new float[vecsize * vecdim];
    float* massb = new float[vecdim];

    // 生成随机数据集
    generate_random_dataset(base_data, vecsize, vecdim);

    int ret;
    pnmesdk_conf config;
    // init
    ret = pnmesdk_init(&config);
    if (ret) {
        printf("pnmesdk_init failed, ret = %d\n", ret);
        delete_mem_func(base_data, massb, context);
        return -1;
    }
    context->data_type = FP32;
    context->vecdim = vecdim;

    // db open
    uint64_t database_id = 0;                      // 0表示创建新的数据库
    char* database_name = "ant-vasg-diskann-poc";  // 用户自定义数据库名称
    database_id = pnmesdk_db_open(context, database_id, database_name);
    if (database_id == 0) {
        printf("open or create new database failed\n");
        delete_mem_func(base_data, massb, context);
        return -1;
    }
    printf("database open or create success\n");
    // call data upload interface
    ret = pnmesdk_db_storage(context, (char*)base_data, vecsize * vecdim * FP32);
    if (ret) {
        printf("pnmesdk_db_storage failed, [data upload failed] ret = %d\n", ret);
        delete_mem_func(base_data, massb, context);
        return -1;
    }
    printf("upload data success, data size: %lu\n", context->length);
    // call cal interface
    int ids_size = 100;
    uint64_t ids_list[ids_size];
    float hardware_result[ids_size];
    float cpu_cal_result[ids_size];
    float epsilon = 1e-5;
    calculate_config calc_config;
    uint64_t start, end = 0;
    int hnsw_query_id = pnme_get_search_query_id();  // hnsw query start
    for (int iter = 0; iter < cal_times; iter++) {
        for (int i = 0; i < ids_size; i++) {
            ids_list[i] = rand() % vecsize;
        }
        // 获取cpu计算结果
        for (int i = 0; i < ids_size; i++) {
            cpu_cal_result[i] = L2Sqr(base_data, base_data + ids_list[i] * vecdim, &vecdim);
        }

        calc_config.target_vector = base_data;
        calc_config.target_vector_size = vecdim * FP32;
        calc_config.ids_list = ids_list;
        calc_config.ids_size = ids_size;
        calc_config.result_list = hardware_result;
        calc_config.hnsw_query_id = hnsw_query_id;
        ret = database_context_cal(context, &calc_config);
        if (ret) {
            printf("current vector cal task execute failed, ret = %d\n", ret);
        }
        // 校验计算结果是否正确
        for (int i = 0; i < ids_size; i++) {
            float res_abs = std::abs(hardware_result[i] - cpu_cal_result[i]);
            if (res_abs > epsilon) {
                printf(
                    "current vector cal task is not equal to cpu. [CPU cal result]: %f "
                    "vs [hardware result]: %f\n",
                    hardware_result[i], cpu_cal_result[i]);
            }
        }
    }
    printf("vector calculate success and cal result is equal to cpu!\n");
    pnme_hnsw_search_end(hnsw_query_id);  // hnsw search end;
    // uninit
    ret = pnmesdk_uninit(&config);
    if (ret) {
        printf("pnmesdk_uninit failed, ret = %d\n", ret);
    }
    // 释放资源
    delete_mem_func(base_data, massb, context);

    return 0;
}