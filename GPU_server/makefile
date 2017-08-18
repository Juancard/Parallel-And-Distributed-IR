CC=gcc
NVCC=nvcc -Wno-deprecated-gpu-targets
MKDIR_P=mkdir -p
BIN_PATH=bin/
OBJ=ir_collection.o ir_collection_handler.o query.o gpu_cuda.o gpu_handler.o docscores.o server_socket.o connection_handler.o init.o

install: $(OBJ) directory
	$(NVCC) -o bin/gpu_server $(OBJ)

directory:
	$(MKDIR_P) $(BIN_PATH)
init.o: init.c gpu_handler.h server_socket.h
	$(NVCC) -o $@ $< -c
server_socket.o: server_socket.c server_socket.h my_socket.h connection_handler.h
	$(NVCC) -o $@ $< -c
connection_handler.o: connection_handler.c connection_handler.h gpu_handler.h
	$(NVCC) -o $@ $< -c
docscores.o: docscores.c docscores.h
	$(NVCC) -o $@ $< -c
gpu_handler.o: gpu_handler.c gpu_handler.h query.h docscores.h ir_collection_handler.h
	$(NVCC) -o $@ $< -c
gpu_cuda.o: gpu_cuda.cu query.h docscores.h ir_collection.h
	$(NVCC) -o $@ $< -c
query.o: query.c query.h
	$(NVCC) -o $@ $< -c
ir_collection_handler.o: ir_collection_handler.c ir_collection_handler.h ir_collection.h
	$(NVCC) -o $@ $< -c
ir_collection.o: ir_collection.c ir_collection.h
	$(NVCC) -o $@ $< -c

clean:
	rm -f *.o
