
#include "local_file_reader.h"


void LocalFileReader::read(std::vector <AlignedRead> &read_reqs, bool async, CallBack callBack) {
    batch_request batch;
    for (int i = 0; i < read_reqs.size(); ++i) {
        batch.emplace_back(read_reqs[i].offset, read_reqs[i].len, read_reqs[i].buf);
    }
    func_(batch, async, callBack);
}

void LocalFileReader::read_and_cal(std::vector<AlignedRead> &read_reqs, bool async, CallBack callBack)
{
#if USE_ALIFLASH == 1
    batch_cal_request batch;
    for (int i = 0; i < read_reqs.size(); ++i) {
        batch.emplace_back(read_reqs[i].neighbor_id, read_reqs[i].dist, read_reqs[i].query);
    }
    func_cal_(batch, async, callBack);
#endif
}

void LocalFileReader::read_and_multi_cal(void *query, uint64_t *ids, float *dists, uint32_t size)
{
#if USE_ALIFLASH == 1
    func_multi_cal_(query, ids, dists, size);
#endif
}

