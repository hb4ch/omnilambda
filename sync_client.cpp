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

const std::string dct8x8_code = "dct8x8.cu\n #include <cooperative_groups.h>\n \n namespace cg = cooperative_groups;\n \n #define BLOCK_SIZE          8\n #define BLOCK_SIZE2         64\n #define BLOCK_SIZE_LOG2     3\n #define BLOCK_SIZE2_LOG2    6\n #define __MUL24_FASTER_THAN_ASTERIX\n #ifdef __MUL24_FASTER_THAN_ASTERIX\n #define FMUL(x,y)   (__mul24(x,y))\n #else\n #define FMUL(x,y)   ((x)*(y))\n #endif\n #define IMAD(a, b, c) ( ((a) * (b)) + (c) )\n #define IMUL(a, b) ((a) * (b))\n \n #define KERS_BLOCK_WIDTH            32\n \n #define KERS_BLOCK_HEIGHT           32\n #define KERS_BW_LOG2                5\n #define KERS_BH_LOG2                5\n #define KERS_SMEMBLOCK_STRIDE       (KERS_BLOCK_WIDTH + 2)\n #define KERS_BLOCK_WIDTH_HALF       (KERS_BLOCK_WIDTH / 2)\n \n #define SIN_1_4     0x5A82\n #define COS_1_4     0x5A82\n #define SIN_1_8     0x30FC\n #define COS_1_8     0x7642\n \n #define OSIN_1_16   0x063E\n #define OSIN_3_16   0x11C7\n #define OSIN_5_16   0x1A9B\n #define OSIN_7_16   0x1F63\n \n #define OCOS_1_16   0x1F63\n #define OCOS_3_16   0x1A9B\n #define OCOS_5_16   0x11C7\n #define OCOS_7_16   0x063E\n typedef union PackedShorts\n {\n     int hShort1;\n     int hShort2;\n     unsigned int hInt;\n } PackedShorts;\n __device__ inline int unfixh(int x)\n {\n     return (int)((x + 0x8000) >> 16);\n }\n __device__ inline int unfixo(int x)\n {\n     return (x + 0x1000) >> 13;\n }\n __device__ void CUDAintInplaceDCT(int *SrcDst, int Stride = 32)\n {\n     int in0, in1, in2, in3, in4, in5, in6, in7;\n     int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;\n     int tmp10, tmp11, tmp12, tmp13;\n     int tmp14, tmp15, tmp16, tmp17;\n     int tmp25, tmp26;\n \n     int DoubleStride = Stride << 1;\n \n     int *DstPtr = SrcDst;\n     in0 = *DstPtr;\n     DstPtr += Stride;\n     in1 = *DstPtr;\n     DstPtr += Stride;\n     in2 = *DstPtr;\n     DstPtr += Stride;\n     in3 = *DstPtr;\n     DstPtr += Stride;\n     in4 = *DstPtr;\n     DstPtr += Stride;\n     in5 = *DstPtr;\n     DstPtr += Stride;\n     in6 = *DstPtr;\n     DstPtr += Stride;\n     in7 = *DstPtr;\n \n     tmp0 = in7 + in0;\n     tmp1 = in6 + in1;\n     tmp2 = in5 + in2;\n     tmp3 = in4 + in3;\n     tmp4 = in3 - in4;\n     tmp5 = in2 - in5;\n     tmp6 = in1 - in6;\n     tmp7 = in0 - in7;\n \n     tmp10 = tmp3 + tmp0;\n     tmp11 = tmp2 + tmp1;\n     tmp12 = tmp1 - tmp2;\n     tmp13 = tmp0 - tmp3;\n \n     tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));\n     tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));\n \n     tmp4 <<= 2;\n     tmp7 <<= 2;\n \n     tmp14 = tmp4 + tmp15;\n     tmp25 = tmp4 - tmp15;\n     tmp26 = tmp7 - tmp16;\n     tmp17 = tmp7 + tmp16;\n \n     DstPtr = SrcDst;\n     *DstPtr = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp10 - tmp11, COS_1_4));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));\n \n     DstPtr = SrcDst + Stride;\n     *DstPtr = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));\n     DstPtr += DoubleStride;\n     *DstPtr = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));\n }\n \n __device__ void CUDAintInplaceDCT(unsigned int *V8)\n {\n     int in0, in1, in2, in3, in4, in5, in6, in7;\n     int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;\n     int tmp10, tmp11, tmp12, tmp13;\n     int tmp14, tmp15, tmp16, tmp17;\n     int tmp25, tmp26;\n     PackedShorts sh0, sh1, sh2, sh3;\n \n     sh0.hInt = V8[0];\n     sh1.hInt = V8[1];\n     sh2.hInt = V8[2];\n     sh3.hInt = V8[3];\n     in0 = sh0.hShort1;\n     in1 = sh0.hShort2;\n     in2 = sh1.hShort1;\n     in3 = sh1.hShort2;\n     in4 = sh2.hShort1;\n     in5 = sh2.hShort2;\n     in6 = sh3.hShort1;\n     in7 = sh3.hShort2;\n \n     tmp0 = in7 + in0;\n     tmp1 = in6 + in1;\n     tmp2 = in5 + in2;\n     tmp3 = in4 + in3;\n     tmp4 = in3 - in4;\n     tmp5 = in2 - in5;\n     tmp6 = in1 - in6;\n     tmp7 = in0 - in7;\n \n     tmp10 = tmp3 + tmp0;\n     tmp11 = tmp2 + tmp1;\n     tmp12 = tmp1 - tmp2;\n     tmp13 = tmp0 - tmp3;\n \n     sh0.hShort1 = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));\n     sh2.hShort1 = unfixh(FMUL(tmp10 - tmp11, COS_1_4));\n \n     sh1.hShort1 = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));\n     sh3.hShort1 = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));\n \n     tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));\n     tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));\n \n     tmp4 <<= 2;\n     tmp7 <<= 2;\n \n     tmp14 = tmp4 + tmp15;\n     tmp25 = tmp4 - tmp15;\n     tmp26 = tmp7 - tmp16;\n     tmp17 = tmp7 + tmp16;\n \n     sh0.hShort2 = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));\n     sh3.hShort2 = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));\n     sh2.hShort2 = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));\n     sh1.hShort2 = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));\n \n     V8[0] = sh0.hInt;\n     V8[1] = sh1.hInt;\n     V8[2] = sh2.hInt;\n     V8[3] = sh3.hInt;\n }\n \n __global__ void CUDAkernelShortDCT(int *SrcDst)\n {\n     int ImgStride = 512;\n     cg::thread_block cta = cg::this_thread_block();\n     __shared__ int block[KERS_BLOCK_HEIGHT * KERS_SMEMBLOCK_STRIDE];\n     int OffsThreadInRow = FMUL(threadIdx.y, BLOCK_SIZE) + threadIdx.x;\n     int OffsThreadInCol = FMUL(threadIdx.z, BLOCK_SIZE);\n     int OffsThrRowPermuted = (OffsThreadInRow & 0xFFFFFFE0) | ((OffsThreadInRow << 1) | (OffsThreadInRow >> 4) & 0x1) & 0x1F;\n \n     SrcDst += IMAD(IMAD(blockIdx.y, KERS_BLOCK_HEIGHT, OffsThreadInCol), ImgStride, IMAD(blockIdx.x, KERS_BLOCK_WIDTH, OffsThreadInRow * 2));\n     int *bl_ptr = block + IMAD(OffsThreadInCol, KERS_SMEMBLOCK_STRIDE, OffsThreadInRow * 2);\n     if (OffsThreadInRow < KERS_BLOCK_WIDTH_HALF)\n     {\n #pragma unroll\n         for (int i = 0; i < BLOCK_SIZE; i++)\n             ((int *)bl_ptr)[i * (KERS_SMEMBLOCK_STRIDE / 2)] = ((int *)SrcDst)[i * (ImgStride / 2)];\n     }\n     cg::sync(cta);\n     CUDAintInplaceDCT(block + OffsThreadInCol * KERS_SMEMBLOCK_STRIDE + OffsThrRowPermuted, KERS_SMEMBLOCK_STRIDE);\n     cg::sync(cta);\n     CUDAintInplaceDCT((unsigned int *)(block + OffsThreadInRow * KERS_SMEMBLOCK_STRIDE + OffsThreadInCol));\n     cg::sync(cta);\n     if (OffsThreadInRow < KERS_BLOCK_WIDTH_HALF)\n     {\n #pragma unroll\n         for (int i = 0; i < BLOCK_SIZE; i++)\n             ((int *)SrcDst)[i * (ImgStride / 2)] = ((int *)bl_ptr)[i * (KERS_SMEMBLOCK_STRIDE / 2)];\n     }\n }\n";

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
    //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
    d.Accept(writer);

    return std::string(strbuf.GetString());
}

std::string create_dct8x8_json()
{
    rapidjson::Document d;
    rapidjson::Document::AllocatorType& alloc = d.GetAllocator();
    d.SetObject();

    rapidjson::Value cuda_code(rapidjson::kObjectType);
    cuda_code.SetString(dct8x8_code.c_str(), dct8x8_code.size());
    d.AddMember("cuda_code", cuda_code, alloc);

    size_t size = 512 * 512;

    rapidjson::Value block_per_grid(rapidjson::kObjectType);
    block_per_grid.SetInt(4096);
    d.AddMember("block_per_grid", block_per_grid, alloc);

    rapidjson::Value threads_per_block(rapidjson::kObjectType);
    threads_per_block.SetInt(64);
    d.AddMember("threads_per_block", threads_per_block, alloc);

    rapidjson::Value data_1(rapidjson::kObjectType);
    rapidjson::Value data_1_type(rapidjson::kObjectType);
    data_1_type.SetString("int");
    data_1.AddMember("type", data_1_type, alloc);

    rapidjson::Value data_1_dim_x(rapidjson::kObjectType);
    data_1_dim_x.SetInt(512);
    data_1.AddMember("dim_x", data_1_dim_x, alloc);

    rapidjson::Value data_1_dim_y(rapidjson::kObjectType);
    data_1_dim_y.SetInt(512);
    data_1.AddMember("dim_y", data_1_dim_y, alloc);

    rapidjson::Value data_1_data(rapidjson::kArrayType);
    srand(0);
    for (size_t i = 0; i < 512*512; i++) {
        data_1_data.PushBack(rand(), alloc);
    }
    data_1.AddMember("data", data_1_data, alloc);
    d.AddMember("data_1", data_1, alloc);

    rapidjson::Value call(rapidjson::kObjectType);
    rapidjson::Value func_name(rapidjson::kObjectType);
    func_name.SetString("CUDAkernelShortDCT");
    call.AddMember("func_name", func_name, alloc);

    rapidjson::Value args(rapidjson::kArrayType);
    args.PushBack("data_1", alloc);
    call.AddMember("args", args, alloc);
    d.AddMember("call", call, alloc);

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
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
        std::cout << boost::beast::buffers(buffer_.data()) << std::endl;
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
    int total_invocation = 400;
    for (int i = 0; i < total_invocation * 2; i++) {
        jsons.emplace_back(create_vectoradd_json(size, size / 10));
        // jsons.emplace_back(create_dct8x8_json());
    }
    int json_i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (total_invocation > 0) {
        
        int request_concur;
        if(lambda > 10) 
            request_concur = poisson_dev(gen);
        else request_concur = lambda;
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

    total_invocation = 400 - total_invocation;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "Total " << total_invocation << " requests" << std::endl;
    std::cout << "Latency: " << duration / (double)total_invocation << " microsecond per requests\n";
    std::cout << "done!" << std::endl;
}
