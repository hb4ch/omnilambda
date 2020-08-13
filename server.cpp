// Omnilamdba Daemon Server

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
// std

#include <boost/asio/bind_executor.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
// boost

#include "scheduler.hpp"
#include "server.hpp"
#include "workload.hpp"

using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
namespace websocket = boost::beast::websocket; // from <boost/beast/websocket.hpp>

void session::run(std::shared_ptr<Scheduler> s)
{
    // Accept the websocket handshake
    ws_.async_accept(boost::asio::bind_executor(
        strand_, std::bind(&session::on_accept, shared_from_this(), s, std::placeholders::_1)));
}

void session::on_accept(std::shared_ptr<Scheduler> s, boost::system::error_code ec)
{
    if (ec)
        return fail(ec, "accept");

    // Read a message
    do_read(s);
}

void session::do_read(std::shared_ptr<Scheduler> s)
{
    // Read a message into our buffer
    ws_.async_read(
        read_buffer_,
        boost::asio::bind_executor(
            strand_, std::bind(&session::on_read, shared_from_this(), s, std::placeholders::_1, std::placeholders::_2)));
}

void session::on_read(std::shared_ptr<Scheduler> s, boost::system::error_code ec,
    std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    // This indicates that the session was closed
    if (ec == websocket::error::closed)
        return;

    if (ec)
        // fail(ec, "read");
        std::cerr << "Reading error from session::on_read\n";

    std::stringstream ss;
    ss << boost::beast::buffers(read_buffer_.data());
    std::shared_ptr<Workload> wl_ptr = std::make_shared<Workload>();
    bool success = wl_ptr->parse(ss.str());
    //wl.output();
    if (success) {
        auto f = std::async(std::launch::async, [s, wl_ptr] {
            s->async_insert_workload(wl_ptr);
        });
    }

    //s->join();

    // {
    //     const std::lock_guard<std::mutex> lock(printlock);
    //     std::cout << boost::beast::buffers(read_buffer_.data()) << std::endl;
    // }
    // Echo the message

    // Now queue the workload and block;

    ws_.text(ws_.got_text());
    std::string ret_json = "echo";

    ws_.async_write(
        boost::asio::buffer(ret_json),
        boost::asio::bind_executor(
            strand_, std::bind(&session::on_write, shared_from_this(), s, std::placeholders::_1, std::placeholders::_2)));
}

void session::on_write(std::shared_ptr<Scheduler> s, boost::system::error_code ec,
    std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    if (ec)
        return fail(ec, "write");
    //std::cerr << "write error\n";

    read_buffer_.consume(read_buffer_.size());

    do_read(s);
}

// Start accepting incoming connections
void listener::run()
{
    if (!acceptor_.is_open())
        return;
    do_accept();
}

void listener::do_accept()
{
    acceptor_.async_accept(socket_,
        std::bind(&listener::on_accept, shared_from_this(),
            std::placeholders::_1));
}

void listener::on_accept(boost::system::error_code ec)
{
    if (ec) {
        fail(ec, "accept");
    } else {
        std::make_shared<session>(std::move(socket_))->run(s_);
    }

    do_accept();
}

int main(int argc, char* argv[])
{
    // Check command line arguments.
    if (argc != 4) {
        std::cerr << "Usage: websocket-server-async <address> <port> <threads>\n"
                  << "Example:\n"
                  << "    websocket-server-async 0.0.0.0 8080 1\n";
        return EXIT_FAILURE;
    }
    auto const address = boost::asio::ip::make_address(argv[1]);
    auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
    auto const threads = std::max<int>(1, std::atoi(argv[3]));

    Workload::global_count = 0;
    // The io_context is required for all I/O
    boost::asio::io_context ioc { threads };
    boost::asio::io_context sched_ioc { 2 };

    std::shared_ptr<Scheduler> s { std::make_shared<Scheduler>(sched_ioc) };
    // Create and launch a listening port
    std::make_shared<listener>(ioc, s, tcp::endpoint { address, port })->run();

    // Run the I/O service on the requested number of threads
    std::vector<std::thread> sched_threads;

    sched_threads.reserve(2);
    sched_threads.emplace_back([&sched_ioc] { sched_ioc.run(); });
    sched_threads.emplace_back([&sched_ioc] { sched_ioc.run(); });

    std::vector<std::thread> v;
    v.reserve(threads - 1);
    for (auto i = threads - 1; i > 0; --i)
        v.emplace_back([&ioc] { ioc.run(); });
    ioc.run();

    return EXIT_SUCCESS;
}