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

const std::string matrixmul_code = "matrixmul \n __global__ void MatrixMul(float *A, float *B, float *C, int w) {\n     int bx = blockIdx.x;\n     int by = blockIdx.y;\n     int tx = threadIdx.x;\n     int ty = threadIdx.y;\n     int aBegin = w * 32 * by;\n     int aEnd   = aBegin + w - 1;\n     int aStep  = 32;\n     int bBegin = 32 * bx;\n     int bStep  = 32 * w;\n     float Csub = 0;\n     for (int a = aBegin, b = bBegin;\n             a <= aEnd;\n             a += aStep, b += bStep) {\n         __shared__ float As[32][32];\n         __shared__ float Bs[32][32];\n \n         As[ty][tx] = A[a + w * ty + tx];\n         Bs[ty][tx] = B[b + w * ty + tx];\n \n         __syncthreads();\n #pragma unroll\n         for (int k = 0; k < 32; ++k) {\n             Csub += As[ty][k] * Bs[k][tx];\n         }\n         __syncthreads();\n     }\n     int c = w * 32 * by + 32 * bx;\n     C[c + w * ty + tx] = Csub;\n }\n ;";

std::string
    create_matrixmul_json(int kernel_size_mean, int kernel_size_delta)
{
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& alloc = d.GetAllocator();
    d.SetObject();

    rapidjson::Value cuda_code(rapidjson::kObjectType);
    cuda_code.SetString(matrixmul_code.c_str(), matrixmul_code.size());
    d.AddMember("cuda_code", cuda_code, alloc);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dev(kernel_size_mean, 0);

    size_t size = normal_dev(gen);
    
    // rapidjson::Value block_per_grid(rapidjson::kObjectType);
    // block_per_grid.SetInt(32 * 32);
    // d.AddMember("block_per_grid", block_per_grid, alloc);

    // rapidjson::Value threads_per_block(rapidjson::kObjectType);
    // threads_per_block.SetInt(size / 32 * size / 32);
    // d.AddMember("threads_per_block", threads_per_block, alloc);

    rapidjson::Value blocks(rapidjson::kArrayType);
    blocks.PushBack(32, alloc);
    blocks.PushBack(32, alloc);
    blocks.PushBack(1, alloc);
    d.AddMember("block_per_grid", blocks, alloc);

    rapidjson::Value threads(rapidjson::kArrayType);
    threads.PushBack(size / 32, alloc);
    threads.PushBack(size / 32, alloc);
    threads.PushBack(1, alloc);
    d.AddMember("threads_per_block", threads, alloc);

    rapidjson::Value data_1(rapidjson::kObjectType);
    rapidjson::Value data_1_type(rapidjson::kObjectType);
    data_1_type.SetString("int");
    data_1.AddMember("type", data_1_type, alloc);

    rapidjson::Value data_1_dim_x(rapidjson::kObjectType);
    data_1_dim_x.SetInt(size);
    data_1.AddMember("dim_x", data_1_dim_x, alloc);

    rapidjson::Value data_1_dim_y(rapidjson::kObjectType);
    data_1_dim_y.SetInt(size);
    data_1.AddMember("dim_y", data_1_dim_y, alloc);

    rapidjson::Value data_1_data(rapidjson::kArrayType);
    srand(0);
    for (size_t i = 0; i < size * size; i++) {
        data_1_data.PushBack(rand() % 1024, alloc);
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
    data_2_dim_y.SetInt(size);
    data_2.AddMember("dim_y", data_2_dim_y, alloc);

    rapidjson::Value data_2_data(rapidjson::kArrayType);
    srand(0);
    for (size_t i = 0; i < size * size; i++) {
        data_2_data.PushBack(0, alloc);
    }
    data_2.AddMember("data", data_2_data, alloc);
    d.AddMember("data_2", data_2, alloc);

    rapidjson::Value result_1(rapidjson::kObjectType);
    rapidjson::Value result_1_type(rapidjson::kObjectType);
    result_1_type.SetString("int");
    result_1.AddMember("type", result_1_type, alloc);

    rapidjson::Value result_1_dim_x;
    result_1_dim_x.SetInt(size);
    rapidjson::Value result_1_dim_y;
    result_1_dim_y.SetInt(size);

    result_1.AddMember("dim_x", result_1_dim_x, alloc);
    result_1.AddMember("dim_y", result_1_dim_y, alloc);

    d.AddMember("result_1", result_1, alloc);

    rapidjson::Value call(rapidjson::kObjectType);
    rapidjson::Value func_name(rapidjson::kObjectType);
    func_name.SetString("MatrixMul");
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
    // rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    d.Accept(writer);

    return std::string(strbuf.GetString());
}


std::string
    create_vectoradd_json(int kernel_size_mean, int kernel_size_delta)
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
    // rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    d.Accept(writer);

    return std::string(strbuf.GetString());
}



using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
namespace websocket = boost::beast::websocket; // from <boost/beast/websocket.hpp>

//------------------------------------------------------------------------------

// Report a failure
void fail(boost::system::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// Sends a WebSocket message and prints the response
class session : public std::enable_shared_from_this<session> {
    tcp::resolver resolver_;
    websocket::stream<tcp::socket> ws_;
    boost::beast::multi_buffer buffer_;
    std::string host_;
    std::string text_;

public:
    // Resolver and socket require an io_context
    explicit session(boost::asio::io_context& ioc)
        : resolver_(ioc)
        , ws_(ioc)
    {
    }

    // Start the asynchronous operation
    void
    run(
        const std::string& host,
        const std::string& port,
        const std::string& text)
    {
        // Save these for later
        host_ = host;
        text_ = text;

        // Look up the domain name
        resolver_.async_resolve(
            host,
            port,
            std::bind(
                &session::on_resolve,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2));
    }

    void
    on_resolve(
        boost::system::error_code ec,
        tcp::resolver::results_type results)
    {
        if (ec)
            return fail(ec, "resolve");

        // Make the connection on the IP address we get from a lookup
        boost::asio::async_connect(
            ws_.next_layer(),
            results.begin(),
            results.end(),
            std::bind(
                &session::on_connect,
                shared_from_this(),
                std::placeholders::_1));
    }

    void
    on_connect(boost::system::error_code ec)
    {
        if (ec)
            return fail(ec, "connect");

        // Perform the websocket handshake
        ws_.async_handshake(host_, "/",
            std::bind(
                &session::on_handshake,
                shared_from_this(),
                std::placeholders::_1));
    }

    void
    on_handshake(boost::system::error_code ec)
    {
        if (ec)
            return fail(ec, "handshake");

        // Send the message
        ws_.async_write(
            boost::asio::buffer(text_),
            std::bind(
                &session::on_write,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2));
    }

    void
    on_write(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return fail(ec, "write");

        // Read a message into our buffer
        ws_.async_read(
            buffer_,
            std::bind(
                &session::on_read,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2));
    }

    void
    on_read(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return fail(ec, "read");

        // Close the WebSocket connection
        ws_.async_close(websocket::close_code::normal,
            std::bind(
                &session::on_close,
                shared_from_this(),
                std::placeholders::_1));
    }

    void
    on_close(boost::system::error_code ec)
    {
        if (ec)
            return fail(ec, "close");

        // If we get here then the connection is closed gracefully

        // The buffers() function helps print a ConstBufferSequence
        // std::cout << boost::beast::buffers(buffer_.data()) << std::endl;
    }
};

void thread_helper(const std::string& host, const std::string& port, const std::string& json)
{
    boost::asio::io_context ioc;
    std::make_shared<session>(ioc)->run(host, port, json);
    ioc.run();
}

// Sends a WebSocket message and prints the response
int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "Usage: sync_client <host> <port> <lambda> <size>\n";
        return EXIT_FAILURE;
    }
    std::string host = argv[1];
    std::string port = argv[2];

    int lambda = atoi(argv[3]);
    int size = atoi(argv[4]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> poisson_dev(lambda);

    std::vector<std::string> jsons;

    // Launch the asynchronous operation
    int total_invocation = 1000;
    for (int i = 0; i < total_invocation * 1.2; i++) {
        jsons.emplace_back(create_matrixmul_json(size, size / 10));
        // jsons.emplace_back(create_dct8x8_json());
    }
    int json_i = 0;
    std::cout << jsons[0];
    std::thread test {std::bind(&thread_helper, std::cref(host), std::cref(port), std::cref(jsons[0]))};
    test.join();
    return 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (total_invocation > 0) {
        
        int request_concur;
        // if(lambda > 10) 
        //     request_concur = poisson_dev(gen);
        // else request_concur = lambda;
        request_concur = lambda;
        //int request_concur = lambda;
        std::cout << "Batch number: " << request_concur << "\n";
        const std::string& json = jsons[json_i++];
        std::vector<std::thread> vec_thread;
        for (int c = 0; c < request_concur; c++) {
            vec_thread.emplace_back(std::bind(&thread_helper, std::cref(host), std::cref(port), std::cref(json)));
        }
        for (int c = 0; c < request_concur; c++) {
            vec_thread[c].join();
        }
        // sleep(1);
        total_invocation -= request_concur;
    }

    total_invocation = 1000 - total_invocation;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // duration -= double(total_invocation);
    std::cout << "Total " << total_invocation << " requests" << std::endl;
    std::cout << "Latency: " << duration / (double)total_invocation << " microsecond per requests\n";
    std::cout << "done!" << std::endl;
}
