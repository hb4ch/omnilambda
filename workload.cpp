#include <exception>
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

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
// rapidjson

#include "workload.hpp"

/*
{
    "cuda_code" : "vectoradd\\n__global__ void\\nvectorAdd(T *A, T *B, T *C, int numElements)\\n{\\nint i = blockDim.x * blockIdx.x + threadIdx.x;\\nif (i < numElements)\\n{\nC[i] = A[i] + B[i];\\nC[i] *= C[i] * 4;\\nC[i] -= 2;\\nC[i] = (C[i]*0.76 + C[i]) * 2;\\n}\\n}\\n",
    "block_per_grid" : 4,
    "threads_per_block" : 16,
    "data_1" : {
        "type" : "int",
        "dim_x" : 4,
        "dim_y" : 4,
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    "data_2" : {
        "type" : "int",
        "dim_x" : 4,
        "dim_y" : 4,
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    "result_1" : {
        "type" : "int",
        "dim_x" : 4,
        "dim_y" : 4
    },
    "call" : {
        "func_name" : "vector_add", 
        "args" : ["data_1", "data_2", "result_1", 16]
    }
}
*/
void Workload::parse(const std::string & json_str)
{
    rapidjson::Document d;
    d.Parse(json_str.c_str());

    cuda_code_ = std::move(d["cuda_code"].GetString());
    block_per_grid_ = d["block_per_grid"].GetInt();
    threads_per_block_ = d["threads_per_block"].GetInt();
    call_func_name_ = std::move(d["call"]["func_name"].GetString());
    const rapidjson::Value& call_args = d["call"]["args"];
    // const char* kTypeNames[] = 
    // { "Null", "False", "True", "Object", "Array", "String", "Number" };
    // std::cout << kTypeNames[call_args.GetType()] << std::endl;
    for (auto& v : call_args.GetArray())
        args_.push_back(v.GetString());

    data_count_ = 0;

    for (const std::string& v : args_) {
        if (v.find("data_") != std::string::npos) {
            Data data;
            std::string type = d[v.c_str()]["type"].GetString();

           
            if (type == "int")
                data.type = Type::INT32;
            else if (type == "long")
                data.type = Type::INT64;
            else if (type == "double")
                data.type = Type::FLOAT64;
            else if (type == "float")
                data.type = Type::FLOAT32;
            else
                throw std::runtime_error("Unknown type of data array.");

            data.dim_x  = d[v.c_str()]["dim_x"].GetUint64();
            data.dim_y = d[v.c_str()]["dim_y"].GetUint64();
            size_t buffer_size = (size_t)data.dim_x  * (size_t)data.dim_y;

            if (data.type == Type::INT32)
                buffer_size *= sizeof(int);
            else if (data.type == Type::INT64)
                buffer_size *= sizeof(long);
            else if (data.type == Type::FLOAT64)
                buffer_size *= sizeof(double);
            else if (data.type == Type::FLOAT32)
                buffer_size *= sizeof(float);

            data.buffer = (void*)malloc(buffer_size + 32);
            data.size = buffer_size + 32;
            void* data_p = data.buffer;
            // Now parses array data_1 ... data_n
            
            const rapidjson::Value & a = d[v.c_str()]["data"];
            assert(a.IsArray());
            for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
                if (data.type == Type::INT32) {
                    *((int*)data_p) = a[i].GetInt();
                    data_p = (int*)data_p + 1;
                } else if (data.type == Type::INT64) {
                    *((int*)data_p) = a[i].GetInt64();
                    data_p = (long*)data_p + 1;
                } else if (data.type == Type::FLOAT32) {
                    *((int*)data_p) = a[i].GetDouble();
                    data_p = (double*)data_p + 1;
                } else if (data.type == Type::FLOAT64) {
                    *((int*)data_p) = a[i].GetDouble();
                    data_p = (float*)data_p + 1;
                }
            }
            data.name = v;
            data_set.push_back(data);
            data_count_++;
            std::cout << "Parsing " << v << std::endl;
        }
        else if (v.find("result_") != std::string::npos) {
            Data res;
            res.name = v;
            std::string type = d[v.c_str()]["type"].GetString();

           
            if (type == "int")
                res.type = Type::INT32;
            else if (type == "long")
                res.type = Type::INT64;
            else if (type == "double")
                res.type = Type::FLOAT64;
            else if (type == "float")
                res.type = Type::FLOAT32;
            else
                throw std::runtime_error("Unknown type of data array.");

            res.dim_x  = d[v.c_str()]["dim_x"].GetUint64();
            res.dim_y = d[v.c_str()]["dim_y"].GetUint64();
            size_t buffer_size = (size_t)res.dim_x  * (size_t)res.dim_y;

            if (res.type == Type::INT32)
                buffer_size *= sizeof(int);
            else if (res.type == Type::INT64)
                buffer_size *= sizeof(long);
            else if (res.type == Type::FLOAT64)
                buffer_size *= sizeof(double);
            else if (res.type == Type::FLOAT32)
                buffer_size *= sizeof(float);

            res.buffer = (void*)malloc(buffer_size);
            res.size = buffer_size;
            std::cout << "Parsing " << v << std::endl;
            result_set.push_back(res);
        }
    }
}

void Workload::output() {
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Parsed data: \n";
    std::cout << "cuda_code: \n" << cuda_code_;
    std::cout << "id: " << id_ << "\n";
    std::cout << "block_per_grid: " << block_per_grid_ << "\n";
    std::cout << "threads_per_block: " << threads_per_block_ << "\n";
    std::cout << "call_func_name: (" << call_func_name_ << ")\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "Data part:\n";
    for(Data& i: data_set) {
        printf("buffer: %p\n", i.buffer);
        printf("dim_x, dim_y = (%d, %d)\n", i.dim_x, i.dim_y);
        std::cout << "data name: " << i.name << "\n";
        std::cout << "data: \n";
        std::cout << "[ ";
        for(size_t j = 0; j < i.size;) {
            if(i.type == Type::INT32) {
                printf("%d ", *((int*)i.buffer + j));
                j += sizeof(int);
            }
            else if(i.type == Type::INT64) {
                printf("%ld ", *((long*)i.buffer + j));
                j += sizeof(long);
            }
            else if(i.type == Type::FLOAT32) {
                printf("%f ", *((float*)i.buffer + j));
                j += sizeof(float);
            }
            else {
                printf("%lf ", *((double*)i.buffer + j));
                j += sizeof(double);
            }
        }
        std::cout << "]\n";
    }
    std::cout << "---------------------------------------------------------\n";
}

void Workload::free() {
    for(Data &d: this->data_set) {
        if(d.buffer)
            std::free(d.buffer);
    }

    for(Data &d: this->result_set) {
        if(d.buffer)
            std::free(d.buffer);
    }
}