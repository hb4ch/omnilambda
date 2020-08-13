#pragma once

#include <vector>
#include <queue>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

template<typename T>
class ts_queue{
private:
    std::queue<T> the_queue;
    mutable boost::mutex the_mutex;
    boost::condition_variable the_condition_variable;

public:
    void push(const T& data) {
        boost::mutex::scoped_lock lock(the_mutex);
        the_queue.push(data);
        lock.unlock();
        the_condition_variable.notify_one();
    }

    bool empty() const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.empty();
    }

    T const &front() const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.front();
    }

    bool try_pop(T& popped_value) {
        boost::mutex::scoped_lock lock(the_mutex);
        if(the_queue.empty())
        {
            return false;
        }
        
        popped_value = the_queue.front();
        the_queue.pop();
        return true;
    }

    void wait_and_pop(T& popped_value) {
        boost::mutex::scoped_lock lock(the_mutex);
        while(the_queue.empty()) {
            the_condition_variable.wait(lock);
        }

        popped_value = the_queue.front();
        the_queue.pop();
    }


    void clear(){ 
        boost::mutex::scoped_lock lock(the_mutex);
        the_queue.clear();
    }

    size_t size() const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.size();
    }
};


template<typename T>
class ts_vector{
private:
    std::vector<T> the_vector;
    mutable boost::mutex the_mutex;

public:
    void push_back(const T& data) {
        boost::mutex::scoped_lock lock(the_mutex);
        the_vector.push_back(data);
    }

    bool empty() const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_vector.empty();
    }

    T const &at(size_t idx) const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_vector[idx];
    }

    void clear(){ 
        boost::mutex::scoped_lock lock(the_mutex);
        the_vector.clear();
    }

    size_t size() const {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_vector.size();
    }
};
