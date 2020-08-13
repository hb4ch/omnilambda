#pragma once

#include <mutex>
#include <queue>
#include <thread>

template<typename T>
class ts_queue{
private:
    std::queue<Data> the_queue;
    mutable std::mutex the_mutex;

public:
    void push(const T& data) {
        std::scoped_lock lock(the_mutex);
        the_queue.push(data);
    }

    bool empty() const {
        std::scoped_lock lock(the_mutex);
        return the_queue.empty();
    }

    T const &front() const {
        std::scoped_lock lock(the_mutex);
        return the_queue.front();
    }

    void pop() {
        std::scoped_lock lock(the_mutex);
        the_queue.pop();
    }
};
