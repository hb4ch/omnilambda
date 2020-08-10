INCLUDE ?= /home/hb4ch/boost_1_68_0/boost/include
LDFLAG ?= -lcuda -lcudart -lnvrtc  
LDBOOST ?= -lboost_system -lpthread
CXXFLAG = -O2 -Wall

all: async_client server

server : server.o workload.o scheduler.o
	g++ server.o workload.o scheduler.o -o server $(LDBOOST)

server.o: server.cpp server.hpp
	ccache g++ $(CXXFLAG) -c server.cpp -I$(INCLUDE)

workload.o: workload.cpp workload.hpp
	g++ $(CXXFLAG) -c workload.cpp -I$(INCLUDE)

scheduler.o : scheduler.cpp scheduler.hpp
	g++ $(CXXFLAG) -c scheduler.cpp -I$(INCLUDE)

async_client : async_client.o
	ccache g++ async_client.o -o async_client $(LDBOOST)

async_client.o: async_client.cpp
	ccache g++ $(CXXFLAG) -c async_client.cpp -I$(INCLUDE)

clean:
	rm -f *.o async_client server