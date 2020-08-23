INCLUDE ?= /home/hb4ch/boost_1_68_0/boost/include
LDCUDA ?= -lcuda -lcudart -lnvrtc  
LDBOOST ?= -lboost_system -lboost_thread -lpthread -pthread
CXXFLAG = -std=c++17 -O2 -Wall
CXX = g++
all: async_client server sync_client

server : server.o workload.o scheduler.o
	$(CXX) server.o workload.o scheduler.o -o server $(LDBOOST) $(LDCUDA)

server.o: server.cpp server.hpp ts_queue.hpp
	$(CXX) $(CXXFLAG) -c server.cpp -I$(INCLUDE)

workload.o: workload.cpp workload.hpp ts_queue.hpp
	$(CXX) $(CXXFLAG) -c workload.cpp -I$(INCLUDE)

scheduler.o : scheduler.cpp scheduler.hpp ts_queue.hpp
	$(CXX) $(CXXFLAG) -c scheduler.cpp -I$(INCLUDE)

async_client : async_client.o
	$(CXX) async_client.o -o async_client $(LDBOOST)

async_client.o: async_client.cpp
	$(CXX) $(CXXFLAG) -c async_client.cpp -I$(INCLUDE)

sync_client: sync_client.cpp
	$(CXX) $(CXXFLAG) sync_client.cpp -I$(INCLUDE) $(LDBOOST) -o sync_client

clean:
	rm -f *.o async_client server sync_client
