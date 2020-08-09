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

struct Data {
    void * buffer;
    int dim_x; 
    int dim_y;
    enum class Type {
        INT32, INT64, FLOAT32, FLOAT64
    };
    Type type;
};

class Workload {
public:
    Workload(uint64_t id) 
        :id_(id)
    {
        
    }
    Workload(const Workload & awl) = default;
    Workload(Workload && awl) = default;
    Workload& operator=(const Workload & awl) = default;
    Workload& operator=(Workload && awl) = default;
    // c'tor

    uint64_t getid() const {
        return id_;
    }
    void parse(const std::string & json_str);
    void run();

    virtual ~Workload() {
        free(return_buf_);
    }

private:
//---------------------------------------------------
    uint64_t id_;
    // const size_t memory_cap = 1024 * 1024 * 1024;

    int block_per_grid_;
    int threads_per_block_;

    std::string cuda_code_;
    std::string call_func_name_;

    std::vector<void*> data_;
    int data_count_;
    
    std::vector<std::string> args_;
    std::vector<std::string> args_type_;

    void * return_buf_;

//-----------------------------------------------------
// static
    std::map<std::string, Data> data_set;
    
};