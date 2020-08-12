
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <thread>

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
namespace websocket = boost::beast::websocket; // from <boost/beast/websocket.hpp>

const std::string vectoradd_code = "vectoradd \n __global__ void \n vectorAdd(int *A, int *B, int *C, int numElements) \n { \n int i = blockDim.x * blockIdx.x + threadIdx.x; \n if (i < numElements)        \n { \n C[i] = A[i] + B[i]; \n C[i] *= C[i] * 4; \n C[i] -= 2; \n C[i] = (C[i]*0.76 + C[i]) * 2;    \n} \n} \n";

std::string create_vectoradd_json(int kernel_size_mean, int kernel_size_delta)
{
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& alloc = d.GetAllocator();
    d.SetObject();

    rapidjson::Value cuda_code(rapidjson::kObjectType);
    cuda_code.SetString(vectoradd_code.c_str(), vectoradd_code.size());
    d.AddMember("cuda_code", cuda_code, alloc);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dev(kernel_size_mean, kernel_size_delta);

    size_t size = normal_dev(gen);
    if (size > 512) {
        rapidjson::Value block_per_grid(rapidjson::kObjectType);
        block_per_grid.SetInt(512);
        d.AddMember("block_per_grid", block_per_grid, alloc);

        rapidjson::Value threads_per_block(rapidjson::kObjectType);
        threads_per_block.SetInt((size + 1) / 512 + 1);
        d.AddMember("threads_per_block", threads_per_block, alloc);
    } else {
        rapidjson::Value block_per_grid(rapidjson::kObjectType);
        block_per_grid.SetInt(1);
        d.AddMember("block_per_grid", block_per_grid, alloc);

        rapidjson::Value threads_per_block(rapidjson::kObjectType);
        threads_per_block.SetInt(size);
        d.AddMember("threads_per_block", threads_per_block, alloc);
    }

    rapidjson::Value data_1(rapidjson::kObjectType);
    rapidjson::Value data_1_type(rapidjson::kObjectType);
    data_1_type.SetString("int");
    data_1.AddMember("type", data_1_type, alloc);

    rapidjson::Value data_1_dim_x(rapidjson::kObjectType);
    data_1_dim_x.SetInt(size);
    data_1.AddMember("dim_x", data_1_dim_x, alloc);

    rapidjson::Value data_1_dim_y(rapidjson::kObjectType);
    data_1_dim_y.SetInt(1);
    data_1.AddMember("dim_y", data_1_dim_y, alloc);

    rapidjson::Value data_1_data(rapidjson::kArrayType);
    srand(0);
    for (size_t i = 0; i < size; i++) {
        data_1_data.PushBack(rand(), alloc);
    }
    data_1.AddMember("data", data_1_data, alloc);
    d.AddMember("data_1", data_1, alloc);

    rapidjson::Value data_2(rapidjson::kObjectType);
    rapidjson::Value data_2_type(rapidjson::kObjectType);
    data_2_type.SetString("int");
    data_2.AddMember("type", data_2_type, alloc);

    rapidjson::Value data_2_dim_x(rapidjson::kObjectType);
    data_2_dim_x.SetInt(size);
    data_2.AddMember("dim_x", data_2_dim_x, alloc);

    rapidjson::Value data_2_dim_y(rapidjson::kObjectType);
    data_2_dim_y.SetInt(1);
    data_2.AddMember("dim_y", data_2_dim_y, alloc);

    rapidjson::Value data_2_data(rapidjson::kArrayType);
    srand(0);
    for (size_t i = 0; i < size; i++) {
        data_2_data.PushBack(rand(), alloc);
    }
    data_2.AddMember("data", data_2_data, alloc);
    d.AddMember("data_2", data_2, alloc);

    rapidjson::Value result_1(rapidjson::kObjectType);
    rapidjson::Value result_1_type(rapidjson::kObjectType);
    result_1_type.SetString("int");
    result_1.AddMember("type", result_1_type, alloc);

    rapidjson::Value result_1_dim_x;
    result_1_dim_x.SetInt(1);
    rapidjson::Value result_1_dim_y;
    result_1_dim_y.SetInt(size);

    result_1.AddMember("dim_x", result_1_dim_x, alloc);
    result_1.AddMember("dim_y", result_1_dim_y, alloc);

    d.AddMember("result_1", result_1, alloc);

    rapidjson::Value call(rapidjson::kObjectType);
    rapidjson::Value func_name(rapidjson::kObjectType);
    func_name.SetString("vectorAdd");
    call.AddMember("func_name", func_name, alloc);

    rapidjson::Value args(rapidjson::kArrayType);
    args.PushBack("data_1", alloc);
    args.PushBack("data_2", alloc);
    args.PushBack("result_1", alloc);
    rapidjson::Value size_str(rapidjson::kObjectType);
    size_str.SetString(std::to_string(size).c_str(), std::to_string(size).size());
    args.PushBack(size_str, alloc);
    call.AddMember("args", args, alloc);
    d.AddMember("call", call, alloc);

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    d.Accept(writer);

    return std::string(strbuf.GetString());
}

int ws_thread(std::string host, std::string port, std::string text)
{
    try {
        boost::asio::io_context ioc;
        tcp::resolver resolver { ioc };
        websocket::stream<tcp::socket> ws { ioc };
        auto const results = resolver.resolve(host, port);
        boost::asio::connect(ws.next_layer(), results.begin(), results.end());
        ws.handshake(host, "/");
        ws.write(boost::asio::buffer(std::string(text)));
        boost::beast::multi_buffer buffer;
        ws.read(buffer);
        ws.close(websocket::close_code::normal);
        std::cout << boost::beast::buffers(buffer.data()) << std::endl;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// Sends a WebSocket message and prints the response
int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: sync_client <host> <port>\n";
        return EXIT_FAILURE;
    }
    std::string host = std::string(argv[1]);
    std::string port = std::string(argv[2]);
    std::string text = create_vectoradd_json(10000, 2000); // size_mean, size_delta

    //std::cout << text;
    std::vector<std::thread> vec_thread;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poisson_dev(7);

    size_t thread_size = poisson_dev(gen);
    for (int test_idx = 0; test_idx < 10; test_idx++) {
        std::cout << "issuing " << thread_size << " client concurrently.\n";
        for (size_t i = 0; i < thread_size; i++) {
            vec_thread.emplace_back(std::bind(&ws_thread, host, port, create_vectoradd_json(20, 10000, 1000)));
        }
        for (size_t i = 0; i < thread_size; i++) {
            vec_thread[i].join();
        }
        
        sleep(0.5);
    }

    //ws_thread(host, port, create_vectoradd_json(20, 10000, 1000));

    std::cout << "done" << std::endl;
}
