#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
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

#define JITIFY_THREAD_SAFE 1
#define JITIFY_PRINT_ALL 1

#include "jitify/cuda_misc.hpp"
#include "jitify/jitify.hpp"
// cuda

#include "scheduler.hpp"
#include "workload.hpp"

// This is threaded.

void Scheduler::ret()
{
    std::unique_lock<std::mutex> lock(ret_mutex_);
    cv_return.wait(lock);
    return;
}

void Scheduler::start()
{
    using namespace std::chrono_literals;
    while (true) {
        std::unique_lock<std::mutex> lock(run_mutex_);
        cv_run.wait_for(lock, largest_timeout_ * 1us);
        async_run();
        cv_return.notify_all();
    }
}

void Scheduler::insert_workload(std::shared_ptr<Workload> wl_ptr)
{
    id_count_++;
    workload_queue_.push(wl_ptr);
    //std::cout << "Inserted workload.\n";

    if (workload_queue_.size() >= (size_t)queue_full_limit_) {
        std::cout << "Queue full...\n";
        timer_.cancel();
        cv_run.notify_all();
        // std::thread t { std::bind(&Scheduler::async_run, this) };
        // t.join();
    }

    if (!queueing_) {
        std::cout << "largest_timeout_: " << largest_timeout_ << std::endl;
        timer_.expires_from_now(boost::posix_time::microseconds(largest_timeout_));
        queueing_ = true;
    }
}

void Scheduler::async_run()
{

    if (workload_queue_.empty())
        return;
    // Batch running...
    queueing_ = false;

    // queue_mutex_.lock();

    std::cout << "Batch running... " << workload_queue_.size() << std::endl;

    if(workload_queue_.size() > prev_queue_size_.load()) {
        largest_timeout_ = std::min(4000L, long(1.4*largest_timeout_));
        prev_queue_size_.store(workload_queue_.size());
    }
    else if(workload_queue_.size() < 0.4 * prev_queue_size_.load()) {
        largest_timeout_ = std::min(4000L, long(0.7*largest_timeout_));
        prev_queue_size_.store(workload_queue_.size());
    }

    while (!workload_queue_.empty()) {
        std::shared_ptr<Workload> front;
        workload_queue_.wait_and_pop(front);
        if (judge_large(front)) {
            process_mode_tasks_.push_back(front);
        } else {
            thread_mode_tasks_.push_back(front);
        }
    }

    // queue_mutex_.unlock();
    //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_prepare)
    // if threads_tasks dominates incoming requests, lower time_out. Otherwise, raise time_out_time;
    
    thread_mode_run();
    process_mode_run();

    thread_mode_tasks_.clear();
    process_mode_tasks_.clear();
}

bool Scheduler::judge_large(std::shared_ptr<Workload> wl_ptr)
{
    long long product = (uint64_t)wl_ptr->get_conf().first * (uint64_t)wl_ptr->get_conf().second;
    std::cout << "first = " << (uint64_t)wl_ptr->get_conf().first << " second = " << (uint64_t)wl_ptr->get_conf().second << "\n";  
    std::cout << "Product = " << product << "\n";
    if (product >= 1500000ULL)
        return true;
    return false;
}

void Scheduler::single_thread(std::shared_ptr<Workload> wl_ptr)
{
    // std::cout << "In child now\n" << std::endl;
    // std::cout << "Running: \n" << wl.cuda_code_;
    //wl_ptr->output();

    static jitify::JitCache kernel_cache;

    //std::cout << wl_ptr->cuda_code_;
    jitify::Program program = kernel_cache.program(wl_ptr->cuda_code_, 0);

    cudaError_t err = cudaSuccess;

    dim3 threadsPerBlock(wl_ptr->threads_per_block_);
    dim3 blocksPerGrid(wl_ptr->block_per_grid_);

    std::vector<void*> data_pointers;
    for (Data& i : wl_ptr->data_set) {
        if (!i.buffer) {
            // queue_mutex_.lock();
            std::cout << "Null data pointers\n";
            // queue_mutex_.unlock();
        }
        data_pointers.push_back(i.buffer);
    }
    std::vector<void*> cuda_pointers;

    for (Data& i : wl_ptr->data_set) {
        void* p = NULL;
        cudaMalloc((void**)&p, i.size);
        cuda_pointers.push_back(p);
    }

    std::vector<void*> cuda_result_pointers;
    for (Data& i : wl_ptr->result_set) {
        void* p = NULL;
        cudaMalloc((void**)&p, i.size);
        cuda_result_pointers.push_back(p);
    }

    for (size_t i = 0; i < data_pointers.size(); i++) {
        cudaMemcpy(cuda_pointers[i], data_pointers[i], wl_ptr->data_set[i].size, cudaMemcpyHostToDevice);
    }

    bool number_arg_exist = 0;
    int number_arg;
    for (std::string& str : wl_ptr->args_) {
        if (std::isdigit(str[0])) {
            number_arg = std::stoi(str);
            number_arg_exist = true;
            break;
        }
    }
    using jitify::reflection::type_of;
    std::cout << "Calling kernel: " << wl_ptr->call_func_name_ << std::endl;
    if (wl_ptr->args_.size() == 4) {
        if (number_arg_exist)
            CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                           .instantiate()
                           .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                           .launch(cuda_pointers[0], cuda_pointers[1], cuda_result_pointers[0], number_arg));
    } else if (wl_ptr->args_.size() == 5) {
        CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                       .instantiate()
                       .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                       .launch(cuda_pointers[0], cuda_pointers[1], cuda_pointers[2], cuda_result_pointers[0], number_arg));
    } else if (wl_ptr->args_.size() == 1) {
        CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                       .instantiate()
                       .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                       .launch(cuda_pointers[0]));
    } else if (wl_ptr->args_.size() == 0) {
        CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                       .instantiate()
                       .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                       .launch(NULL));
    } else {
        std::cout << "Unsupported kernel" << std::endl;
    }

    for (size_t i = 0; i < wl_ptr->result_set.size(); i++) {
        cudaMemcpy(wl_ptr->result_set[i].buffer, cuda_result_pointers[i], wl_ptr->result_set[i].size, cudaMemcpyDeviceToHost);
    }

    for (void* p : cuda_pointers) {
        cudaFree(p);
    }

    for (void* p : cuda_result_pointers) {
        cudaFree(p);
    }
}

void Scheduler::process_mode_run()
{
    // queue_mutex_.lock();
    int n = process_mode_tasks_.size();
    // queue_mutex_.unlock();

    if (n == 0) {
        // std::cout << "No process_mode tasks, returning...\n";
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

    size_t process_num = this->process_mode_tasks_.size();
    for (size_t i = 0; i < process_num; i++) {
        pid_t forked = fork();
        //std::cout << "Forked = " << forked << "\n";
        if ((pids[i] = forked) < 0) {
            perror("fork");
            abort();
        } else if (forked == 0) {
            std::cout << "i'm a child\n";
            //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_child);
            single_thread(process_mode_tasks_.at(i));
            exit(0);
        }
    }

    // if (getpid() == parent_pid) {
    //     //timer_.get_io_service().notify_fork(boost::asio::io_service::fork_parent);
    // }

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
    // if (thread_mode_tasks_.size() == 0) {
    //     std::cout << "No thread_mode tasks, returning...\n";
    //     return;
    // }
    std::cout << "thread_mode_starting...\n";
    int thread_num = thread_mode_tasks_.size();
    //queue_mutex_.lock();
    std::cout << "thread_num = " << thread_num;
    //queue_mutex_.unlock();
    std::vector<std::thread> cuda_threads;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        cuda_threads.emplace_back([this, thread_idx]() {
            std::shared_ptr<Workload> wl_ptr = thread_mode_tasks_.at(thread_idx);
            // std::cout << "Running: \n" << wl.cuda_code_;
            // wl_ptr->output();
            cudaSetDevice(0);
            // This to avoid subtle multithreading-bugs;
            // see https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
            static jitify::JitCache kernel_cache;
            jitify::Program program = kernel_cache.program(wl_ptr->cuda_code_, 0);

            cudaError_t err = cudaSuccess;

            // dim3 threadsPerBlock(wl_ptr->threads_per_block_);
            // dim3 blocksPerGrid(wl_ptr->block_per_grid_);

            std::vector<void*> data_pointers;
            for (Data& i : wl_ptr->data_set) {
                if (!i.buffer) {
                    std::cout << "Null data pointers\n";
                }
                data_pointers.push_back(i.buffer);
            }
            std::vector<void*> cuda_pointers;

            for (Data& i : wl_ptr->data_set) {
                void* p = NULL;
                err = cudaMalloc((void**)&p, i.size);
                if (err) {
                    std::cerr << "cudaMalloc error!\n";
                    return;
                }
                cuda_pointers.push_back(p);
            }

            std::vector<void*> cuda_result_pointers;
            for (Data& i : wl_ptr->result_set) {
                void* p = NULL;

                err = cudaMalloc((void**)&p, i.size);
                if (err) {
                    std::cerr << "cudaMalloc error!\n";
                    return;
                }

                cuda_result_pointers.push_back(p);
            }

            for (size_t i = 0; i < data_pointers.size(); i++) {
                err = cudaMemcpy(cuda_pointers[i], data_pointers[i], wl_ptr->data_set[i].size, cudaMemcpyHostToDevice);
                if (err) {
                    std::cerr << "cudaMemcpy error!\n";
                    return;
                }
            }

            bool number_arg_exist = 0;
            int number_arg;
            for (std::string& str : wl_ptr->args_) {
                if (std::isdigit(str[0])) {
                    number_arg = std::stoi(str);
                    number_arg_exist = true;
                    break;
                }
            }
            using jitify::reflection::type_of;
            if (wl_ptr->args_.size() == 4) {
                if (number_arg_exist)
                    CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                                   .instantiate()
                                   .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                                   .launch(cuda_pointers[0], cuda_pointers[1], cuda_result_pointers[0], number_arg));
            } else if (wl_ptr->args_.size() == 5) {
                CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                               .instantiate()
                               .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                               .launch(cuda_pointers[0], cuda_pointers[1], cuda_pointers[2], cuda_result_pointers[0], number_arg));
            } else if (wl_ptr->args_.size() == 1) {
                CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                               .instantiate()
                               .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                               .launch(cuda_pointers[0]));
            } else if (wl_ptr->args_.size() == 0) {
                CHECK_CUDA(program.kernel(wl_ptr->call_func_name_)
                        .instantiate()
                        .configure(wl_ptr->block_per_grid_, wl_ptr->threads_per_block_)
                        .launch(NULL));                  
            } else {
                std::cout << "Unsupported kernel" << std::endl;
            }

            for (size_t i = 0; i < wl_ptr->result_set.size(); i++) {
                err = cudaMemcpy(wl_ptr->result_set[i].buffer, cuda_result_pointers[i], wl_ptr->result_set[i].size, cudaMemcpyDeviceToHost);
                if (err) {
                    std::cerr << "cudaMemcpy back error! errno: " << err << "\n";
                }
            }

            for (void* p : cuda_pointers) {
                if (!p) {
                    std::cerr << "Trying to free null cuda ptrs. (l. 344)\n";
                }
                err = cudaFree(p);
                if (err) {
                    std::cerr << "cudaMalloc error! errno: "  << err << "\n";
                }
            }

            for (void* p : cuda_result_pointers) {
                if (!p) {
                    std::cerr << "Trying to free null cuda ptrs. (l. 354)\n";
                }
                err = cudaFree(p);
                if (err) {
                    std::cerr << "cudaFree error! errno: " << err << "\n";
                }
            }
        });
    }
    for (auto& i : cuda_threads) {
        i.join();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    std::cout << "Thread mode run time: " << duration << " seconds\n";
}
