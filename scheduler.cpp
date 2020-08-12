#include <chrono>
#include <deque>
#include <map>
#include <string>
#include <thread>
#include <vector>
// std

#include <boost/asio.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/strand.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
// boost

#include <sys/wait.h>
#include <unistd.h>
// posix

#include "jitify/cuda_misc.hpp"
#include "jitify/jitify.hpp"
// cuda

#include "scheduler.hpp"
#include "workload.hpp"

#define JITIFY_THREAD_SAFE 1

void Scheduler::async_insert_workload(Workload&& wl)
{
    queue_mutex_.lock();
    id_count_++;
    workload_queue_.push_back(wl.getid());
    std::cout << "Inserted workload.\n";
    map_id_workload_.emplace(std::make_pair(wl.getid(), std::move(wl)));
    queue_mutex_.unlock();
    if (!queueing_) {
        timer_.expires_from_now(boost::posix_time::seconds(2));
        queueing_ = true;
        std::async(std::launch::async, [this] {
            timer_.wait();
            time_out_ = true;
            queueing_ = false;
            async_run();
        });
    }

    if (workload_queue_.size() == (size_t)queue_full_limit_) {
        std::cout << "Queue full...\n";
        timer_.cancel();
        std::async(std::launch::async, std::bind(&Scheduler::async_run, this));
    }
}

void Scheduler::async_run()
{
    // Batch running...
    queueing_ = false;

    queue_mutex_.lock();
    std::cout << "Batch running... " << workload_queue_.size() << std::endl;
    for (int wl_n : workload_queue_) {
        if (judge_large(wl_n)) {
            process_mode_tasks_.push_back(wl_n);
        } else
            thread_mode_tasks_.push_back(wl_n);
    }
    workload_queue_.clear();
    queue_mutex_.unlock();
    //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_prepare);

    thread_mode_run();

    queue_mutex_.lock();
    thread_mode_tasks_.clear();
    queue_mutex_.unlock();
    
    process_mode_run();

    queue_mutex_.lock();
    process_mode_tasks_.clear();
    queue_mutex_.unlock();

    map_id_workload_.clear();
}

bool Scheduler::judge_large(int wl_n)
{
    Workload& curr = map_id_workload_[wl_n];
    if ((uint64_t)curr.get_conf().first * (uint64_t)curr.get_conf().second >= 1000000ULL)
        return true;

    //return false; TODO:
    return true;
}

void Scheduler::single_thread(int thread_idx)
{
    std::cout << "In child now\n" << std::endl;
    queue_mutex_.lock();
    int task_id = process_mode_tasks_[thread_idx];
    Workload wl = map_id_workload_[task_id];
    std::cout << "Running: \n" << wl.cuda_code_;
    wl.output();
    queue_mutex_.unlock();

    jitify::JitCache kernel_cache;

    std::cout << "About to compile: \n";
    std::cout << wl.cuda_code_;
    jitify::Program program = kernel_cache.program(wl.cuda_code_, 0);

    cudaError_t err = cudaSuccess;

    dim3 threadsPerBlock(wl.threads_per_block_);
    dim3 blocksPerGrid(wl.block_per_grid_);

    Type type = wl.data_set[0].type;

    std::vector<void*> data_pointers;
    for (Data& i : wl.data_set) {
        if (!i.buffer) {
            queue_mutex_.lock();
            std::cout << "Null data pointers\n";
            queue_mutex_.unlock();
        }
        data_pointers.push_back(i.buffer);
    }
    std::vector<void*> cuda_pointers;

    for (Data& i : wl.data_set) {
        void* p = NULL;
        cudaMalloc((void**)&p, i.size);
        cuda_pointers.push_back(p);
    }

    std::vector<void*> cuda_result_pointers;
    for (Data& i : wl.result_set) {
        void* p = NULL;
        cudaMalloc((void**)&p, i.size);
        cuda_result_pointers.push_back(p);
    }

    for (size_t i = 0; i < data_pointers.size(); i++) {
        cudaMemcpy(cuda_pointers[i], data_pointers[i], wl.data_set[i].size, cudaMemcpyHostToDevice);
    }

    bool number_arg_exist = 0;
    int number_arg;
    for (std::string& str : wl.args_) {
        if (std::isdigit(str[0])) {
            number_arg = std::stoi(str);
            number_arg_exist = true;
            break;
        }
    }
    using jitify::reflection::type_of;
    std::cout << "Calling kernel: " << wl.call_func_name_ << std::endl;
    if (wl.args_.size() == 4) {
        if (number_arg_exist)
            CHECK_CUDA(program.kernel(wl.call_func_name_)
                           .instantiate()
                           .configure(blocksPerGrid, threadsPerBlock)
                           .launch(cuda_pointers[0], cuda_pointers[1], cuda_result_pointers[0], number_arg));
    } else if (wl.args_.size() == 5) {
        CHECK_CUDA(program.kernel(wl.call_func_name_)
                       .instantiate()
                       .configure(blocksPerGrid, threadsPerBlock)
                       .launch(cuda_pointers[0], cuda_pointers[1], cuda_pointers[2], cuda_result_pointers[0], number_arg));
    } else {
        std::cout << "Unsupported kernel" << std::endl;
    }

    for (size_t i = 0; i < wl.result_set.size(); i++) {
        cudaMemcpy(wl.result_set[i].buffer, cuda_result_pointers[i], wl.result_set[i].size, cudaMemcpyDeviceToHost);
    }

    for (void* p : cuda_pointers) {
        cudaFree(p);
    }

    for (void* p : cuda_result_pointers) {
        cudaFree(p);
    }

    for (void* p : data_pointers) {
        free(p);
    }

}

void Scheduler::process_mode_run()
{
    queue_mutex_.lock();
    int n = process_mode_tasks_.size();
    queue_mutex_.unlock();

    if (n == 0) {
        std::cout << "No process_mode tasks, returning...\n";
        return;
    }
    std::cout << "process_mode_starting...\n";
    std::array<pid_t, 256> pids;
    int i;

    //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_prepare);
    pid_t parent_pid = getpid();
    /* Start children. */
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "getpid called.\n";
    for (i = 0; i < n; ++i) {
        pid_t forked = fork();
        std::cout << "Forked = " << forked << "\n";
        if ((pids[i] = forked) < 0) {
            perror("fork");
            abort();
        } else if (forked == 0) {
            std::cout << "i'm a child\n";
            //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_child);
            single_thread(i);
            exit(0);
        }
    }
    if (getpid() == parent_pid) {
        //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_parent);
    }

    int status;
    pid_t pid;
    int count = n;
    while (count > 0) {
        pid = wait(&status);
        --count; // TODO(pts): Remove pid from the pids array.
    }

    if (getpid() == parent_pid) {
        //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_parent);
        std::cout << "Process mode done!\n";
    }
    
}

void Scheduler::thread_mode_run()
{
    if (thread_mode_tasks_.size() == 0) {
        std::cout << "No thread_mode tasks, returning...\n";
        return;
    }
    std::cout << "thread_mode_starting...\n";
    int thread_num = thread_mode_tasks_.size();
    std::cout << "thread_num = " << thread_num;
    std::vector<std::thread> cuda_threads;
    for (int thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        cuda_threads.emplace_back([this, thread_idx]() {
            queue_mutex_.lock();
            int task_id = thread_mode_tasks_[thread_idx];
            Workload wl = map_id_workload_[task_id];
            // std::cout << "Running: \n" << wl.cuda_code_;
            wl.output();
            queue_mutex_.unlock();

            jitify::JitCache kernel_cache;

            std::cout << "About to compile: \n";
            std::cout << wl.cuda_code_;
            jitify::Program program = kernel_cache.program(wl.cuda_code_, 0);

            cudaError_t err = cudaSuccess;

            dim3 threadsPerBlock(wl.threads_per_block_);
            dim3 blocksPerGrid(wl.block_per_grid_);

            Type type = wl.data_set[0].type;

            std::vector<void*> data_pointers;
            for (Data& i : wl.data_set) {
                if (!i.buffer) {
                    queue_mutex_.lock();
                    std::cout << "Null data pointers\n";
                    queue_mutex_.unlock();
                }
                data_pointers.push_back(i.buffer);
            }
            std::vector<void*> cuda_pointers;

            for (Data& i : wl.data_set) {
                void* p = NULL;
                cudaMalloc((void**)&p, i.size);
                cuda_pointers.push_back(p);
            }

            std::vector<void*> cuda_result_pointers;
            for (Data& i : wl.result_set) {
                void* p = NULL;
                cudaMalloc((void**)&p, i.size);
                cuda_result_pointers.push_back(p);
            }

            for (size_t i = 0; i < data_pointers.size(); i++) {
                cudaMemcpy(cuda_pointers[i], data_pointers[i], wl.data_set[i].size, cudaMemcpyHostToDevice);
            }

            bool number_arg_exist = 0;
            int number_arg;
            for (std::string& str : wl.args_) {
                if (std::isdigit(str[0])) {
                    number_arg = std::stoi(str);
                    number_arg_exist = true;
                    break;
                }
            }
            std::cout << "here" << std::endl;
            using jitify::reflection::type_of;
            std::cout << "Calling kernel: " << wl.call_func_name_ << std::endl;
            if (wl.args_.size() == 4) {
                if (number_arg_exist)
                    CHECK_CUDA(program.kernel(wl.call_func_name_)
                                   .instantiate()
                                   .configure(blocksPerGrid, threadsPerBlock)
                                   .launch(cuda_pointers[0], cuda_pointers[1], cuda_result_pointers[0], number_arg));
            } else if (wl.args_.size() == 5) {
                CHECK_CUDA(program.kernel(wl.call_func_name_)
                               .instantiate()
                               .configure(blocksPerGrid, threadsPerBlock)
                               .launch(cuda_pointers[0], cuda_pointers[1], cuda_pointers[2], cuda_result_pointers[0], number_arg));
            } else {
                std::cout << "Unsupported kernel" << std::endl;
            }

            for (size_t i = 0; i < wl.result_set.size(); i++) {
                cudaMemcpy(wl.result_set[i].buffer, cuda_result_pointers[i], wl.result_set[i].size, cudaMemcpyDeviceToHost);
            }

            for (void* p : cuda_pointers) {
                cudaFree(p);
            }

            for (void* p : cuda_result_pointers) {
                cudaFree(p);
            }

            for (void* p : data_pointers) {
                free(p);
            }
        });
    }
    for (auto& i : cuda_threads) {
        i.join();
    }

    std::cout << "Thread mode running done.\n";
}
