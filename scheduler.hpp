#pragma once

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

class Scheduler {

    std::deque<int> workload_queue_;
    std::vector<int> thread_mode_tasks_;
    std::vector<int> process_mode_tasks_;

    std::map<int, Workload> map_id_workload_;
    // The actual container of these;
    std::mutex queue_lock_;

    int queue_full_limit_;
    int largest_timeout_;
    // params;
    int judge_workload();
    int readjust_param();

    bool isTasksRunning_;

public:
    Scheduler() {

        // Setting up nvrtc stuff..
    }
    void async_insert_workload(Workload && wl);
    void async_run();
    void join_tasks(); // blocking all tasks until finished and dispatched.
    void join_insertion(); // block all insertion tasks, which rarely happens
    
};