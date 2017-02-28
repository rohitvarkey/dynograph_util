#include "alg_data_manager.h"
#include "logger.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

using namespace DynoGraph;

AlgDataManager::AlgDataManager(int64_t nv, std::vector<std::string> alg_names)
{
    for (std::string alg_name : alg_names)
    {
        last_epoch_data[alg_name].resize(nv);
        current_epoch_data[alg_name].resize(nv);
    }

    path = "";
    if (const char* filename = getenv("DYNOGRAPH_ALG_DATA_PATH"))
    {
        path = std::string(filename);
    }
}

void
AlgDataManager::next_epoch()
{
    // "last" <= "current" for each alg
    for (auto entry : current_epoch_data)
    {
        std::string alg_name = entry.first;
        auto& last = last_epoch_data[alg_name];
        auto& current = current_epoch_data[alg_name];
        last.assign(current.begin(), current.end());
    }
}

void
AlgDataManager::rollback()
{
    // "current" <= "last" for each alg
    for (auto entry : current_epoch_data)
    {
        std::string alg_name = entry.first;
        auto& last = last_epoch_data[alg_name];
        auto& current = current_epoch_data[alg_name];
        current.assign(last.begin(), last.end());
    }
}

void
AlgDataManager::dump(int64_t epoch) const
{
    if (path.empty()) { return; }

    // Create directory for this epoch
    std::string path_root = path + "/" + std::to_string(epoch);
    mkdir(path_root.c_str(), 0700);

    // Write current epoch results to disk
    for (auto entry : current_epoch_data)
    {
        int64_t *ptr = entry.second.data();
        size_t sz = entry.second.size();
        std::string alg_name = entry.first;
        std::string full_path = path_root + "/" + alg_name;
        FILE* fp = fopen(full_path.c_str(), "wb");
        if (fp) {
            fwrite(ptr, sizeof(int64_t), sz, fp);
            fclose(fp);
        } else {
            DynoGraph::Logger::get_instance() << "WARNING: Unable to dump alg results to " << full_path << "\n";
        }
    }
}

std::vector<int64_t>&
AlgDataManager::get_data_for_alg(std::string alg_name) {
    return current_epoch_data.at(alg_name);
}
