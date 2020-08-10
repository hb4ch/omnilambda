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
    "cuda_code" : "....",
    "block_per_grid" : 5,
    "threads_per_block" : 1024,
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
    "call" : {
        "func_name" : "vector_add", 
        "args" : ["data_1", "data_2"]
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
            void* data_p = data.buffer;
            // Now parses array data_1 ... data_n
            std::cout << "Parsing " << v << std::endl;
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
            
            data_set[v] = data;
        }
    }
}