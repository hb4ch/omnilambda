#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <chrono>
#include <map>
// std

#include <boost/asio.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
// boost

#include "workload.hpp"
#include "scheduler.hpp"

void Scheduler::async_insert_workload(Workload&& wl)
{
    queue_mutex_.lock();
    id_count_++;
    workload_queue_.push_back(wl.getid());
    std::cout << "Inserted workload.\n";
    map_id_workload_.emplace(std::make_pair(wl.getid(), std::move(wl)));
    queue_mutex_.unlock();
    if(!queueing_) {
        timer_.expires_from_now(boost::posix_time::seconds(2));
        queueing_ = true;
        std::async(std::launch::async, [this] {
            timer_.wait();
            time_out_ = true;
            queueing_ = false;
            async_run();
        });
    }
    
    if(workload_queue_.size() == (size_t)queue_full_limit_) {
        std::cout << "Queue full...\n";
        timer_.cancel();
        std::async(std::launch::async, std::bind(&Scheduler::async_run, this));
    }
}

void Scheduler::async_run() {
    // Batch running...
    queueing_ = false;

    std::cout << "Batch running... " << workload_queue_.size() << std::endl;

}