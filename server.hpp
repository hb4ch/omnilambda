#pragma once

#include <boost/asio/bind_executor.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
// boost::beast

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
// std

#include "workload.hpp"
#include "scheduler.hpp"

using tcp = boost::asio::ip::tcp; // from <boost/asio/ip/tcp.hpp>
namespace websocket = boost::beast::websocket; // from <boost/beast/websocket.hpp>


void fail(boost::system::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// Echoes back all received WebSocket messages
class session : public std::enable_shared_from_this<session> {
    
    std::mutex printlock;
    websocket::stream<tcp::socket> ws_;
    boost::asio::strand<
        boost::asio::io_context::executor_type>
        strand_;
    boost::beast::multi_buffer read_buffer_;

    uint64_t wl_count = 0;
    // omnilambda workload

public:
    // Take ownership of the socket
    explicit session(tcp::socket socket)
        : ws_(std::move(socket))
        , strand_(ws_.get_executor())
    {
        wl_count++;
    }

    // Start the asynchronous operation
    void run(std::shared_ptr<Scheduler> s);
    void on_accept(std::shared_ptr<Scheduler> s, boost::system::error_code ec);
    void do_read(std::shared_ptr<Scheduler> s);
    void on_read(
        std::shared_ptr<Scheduler> s,
        boost::system::error_code ec,
        std::size_t bytes_transferred);
    void on_write(
        std::shared_ptr<Scheduler> s,
        boost::system::error_code ec,
        std::size_t bytes_transferred);
};

//------------------------------------------------------------------------------

// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener> {
    tcp::acceptor acceptor_;
    tcp::socket socket_;

    std::shared_ptr<Scheduler> s_;
public:
    listener(
        boost::asio::io_context& ioc,
        std::shared_ptr<Scheduler> s,
        tcp::endpoint endpoint)
        : acceptor_(ioc)
        , socket_(ioc)
        , s_(s)
    {
        boost::system::error_code ec;

        // Open the acceptor
        acceptor_.open(endpoint.protocol(), ec);
        if (ec) {
            fail(ec, "open");
            return;
        }

        // Allow address reuse
        acceptor_.set_option(boost::asio::socket_base::reuse_address(true), ec);
        if (ec) {
            fail(ec, "set_option");
            return;
        }

        // Bind to the server address
        acceptor_.bind(endpoint, ec);
        if (ec) {
            fail(ec, "bind");
            return;
        }

        // Start listening for connections
        acceptor_.listen(
            boost::asio::socket_base::max_listen_connections, ec);
        if (ec) {
            fail(ec, "listen");
            return;
        }
    }

    // Start accepting incoming connections
    void run();
    void do_accept();
    void on_accept(boost::system::error_code ec);
};
