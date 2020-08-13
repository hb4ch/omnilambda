#pragma once

#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <chrono>
#include <memory>
#include <map>
#include <atomic>

// std
#include <boost/asio/strand.hpp>
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
// boost

#include "workload.hpp"

class Scheduler : public std::enable_shared_from_this<Scheduler> {

    std::vector<int> workload_queue_;
    std::vector<int> thread_mode_tasks_;
    std::vector<int> process_mode_tasks_;

    std::map<int, Workload> map_id_workload_;
    // The actual container of these;

    int queue_full_limit_;
    long largest_timeout_;
    // params;
    int readjust_param();

    std::mutex queue_mutex_;
    std::mutex cuda_mutex_;
    bool isTasksRunning_;
    std::atomic<bool> time_out_;
    boost::asio::io_service::strand queue_strand_;
    boost::asio::io_service::strand timer_strand_;
    boost::asio::deadline_timer timer_;
    uint64_t id_count_;
    std::atomic<bool> queueing_;


public:

    Scheduler(boost::asio::io_context & sched_ioc)
        : queue_full_limit_(24),
          largest_timeout_(500),
          time_out_(false),
          queue_strand_(sched_ioc),
          timer_strand_(sched_ioc),
          timer_(sched_ioc),
          id_count_(0ULL),
          queueing_(false)
    {
        std::cout << "Scheduler initiated.\n";
    }
    uint64_t get_id_count() {
        return id_count_;
    }
    void start() {
        std::cout << "Context running\n";
    }
    void async_insert_workload(Workload && wl);
    void async_run();
    bool judge_large(int wl_n);
    void single_thread(int thread_idx);
    void thread_mode_run();
    void process_mode_run();
    void join_tasks(); // blocking all tasks until finished and dispatched.
    void join_insertion(); // block all insertion tasks, which rarely happens
};