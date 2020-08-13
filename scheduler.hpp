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
#include "ts_queue.hpp"

class Scheduler : public std::enable_shared_from_this<Scheduler> {

    //std::vector<int> workload_queue_;
    ts_queue<std::shared_ptr<Workload>> workload_queue_;
    ts_vector<std::shared_ptr<Workload>> thread_mode_tasks_;
    ts_vector<std::shared_ptr<Workload>> process_mode_tasks_;

    //std::map<int, Workload> map_id_workload_;
    //ts_map<int, Workload> map_id_workload_;
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
          largest_timeout_(5000),
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
    void async_insert_workload(std::shared_ptr<Workload>);
    void async_run();
    bool judge_large(std::shared_ptr<Workload>);
    void single_thread(std::shared_ptr<Workload>);
    void thread_mode_run();
    void process_mode_run();
    void join_tasks(); // blocking all tasks until finished and dispatched.
    void join_insertion(); // block all insertion tasks, which rarely happens
};