#pragma once

#include <string>
#include <vector>
// std

#include <boost/asio.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
// boost
enum class Type {
    INT32, INT64, FLOAT32, FLOAT64
};

struct Data {
    std::string name;
    void * buffer;
    int dim_x; 
    int dim_y;
    size_t size;
    Type type;
};

class Workload {
public:
    Workload() 
        :id_(this->global_count)
    {
        this->global_count++;   
    }
    Workload(const Workload & awl) = default;
    Workload(Workload && awl) = default;
    Workload& operator=(const Workload & awl) = default;
    Workload& operator=(Workload && awl) = default;
    // c'tor

    uint64_t getid() const {
        return id_;
    }
    std::pair<int, int> get_conf() {
        return {block_per_grid_, threads_per_block_};
    }
    bool parse(const std::string & json_str);
    void output();
    void run();
    void free();

    std::string cuda_code_;
    uint64_t id_;

    int block_per_grid_;
    int threads_per_block_;

    std::string call_func_name_;

    std::vector<void*> data_;
    int data_count_;
    
    std::vector<std::string> args_;
    std::vector<std::string> args_type_;

    void * return_buf_;
    inline static std::atomic<uint64_t> global_count;

    std::vector<Data> data_set;
    std::vector<Data> result_set;

    
};