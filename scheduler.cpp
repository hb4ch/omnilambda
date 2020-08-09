#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <chrono>
#include <map>
// std

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
// boost

#include "workload.hpp"
#include "scheduler.hpp"

void Scheduler::async_insert_workload(Workload&& wl)
{
    std::async(std::launch::async, [this, wl] {
        if(map_id_workload_.find(wl.getid()) != map_id_workload_.end()) {
            assert(false);
        }
        workload_queue_.push_back(wl.getid());
        map_id_workload_.emplace(std::make_pair(wl.getid(), std::move(wl)));
    });
}
