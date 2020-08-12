INCLUDE ?= /home/hb4ch/boost_1_68_0/boost/include
LDCUDA ?= -lcuda -lcudart -lnvrtc  
LDBOOST ?= -lboost_system -lpthread -pthread
CXXFLAG = -std=c++17 -O2 -Wall

all: async_client server sync_client

server : server.o workload.o scheduler.o
	g++ server.o workload.o scheduler.o -o server $(LDBOOST) $(LDCUDA)

server.o: server.cpp server.hpp
	g++ $(CXXFLAG) -c server.cpp -I$(INCLUDE)

workload.o: workload.cpp workload.hpp
	g++ $(CXXFLAG) -c workload.cpp -I$(INCLUDE)

scheduler.o : scheduler.cpp scheduler.hpp
	g++ $(CXXFLAG) -c scheduler.cpp -I$(INCLUDE)

async_client : async_client.o
	g++ async_client.o -o async_client $(LDBOOST)

async_client.o: async_client.cpp
	g++ $(CXXFLAG) -c async_client.cpp -I$(INCLUDE)

sync_client: sync_client.cpp
	g++ $(CXXFLAG) sync_client.cpp -I$(INCLUDE) $(LDBOOST) -o sync_client

clean:
	rm -f *.o async_client server sync_client