CC=gcc
NVCC=nvcc -Wno-deprecated-gpu-targets
MKDIR_P=mkdir -p
BIN_PATH=bin/
OBJ=ir_collection.o ir_collection_handler.o query.o docscores.o server_socket.o connection_handler.o init.o
CUDA_OBJ=gpu_handler.o gpu_cuda.o $(OBJ)
CUDA_SIMULATION_OBJ=gpu_handler_simulation.o $(OBJ)

install: $(CUDA_OBJ) directory
	$(NVCC) -o bin/gpu_server $(CUDA_OBJ)
gpu_simulation: $(CUDA_SIMULATION_OBJ) directory
	$(CC) -o bin/gpu_server_simulation $(CUDA_SIMULATION_OBJ)

directory:
	$(MKDIR_P) $(BIN_PATH)

init.o: init.c gpu_handler.h server_socket.h
	$(CC) -o $@ $< -c
server_socket.o: server_socket.c server_socket.h my_socket.h connection_handler.h
	$(CC) -o $@ $< -c
connection_handler.o: connection_handler.c connection_handler.h gpu_handler.h my_socket.h docscores.h
	$(CC) -o $@ $< -c
docscores.o: docscores.c docscores.h
	$(CC) -o $@ $< -c
gpu_handler.o: gpu_handler.c gpu_handler.h query.h docscores.h ir_collection_handler.h
	$(CC) -o $@ $< -c
gpu_cuda.o: gpu_cuda.cu cuda_error_handler.cu query.h docscores.h ir_collection.h
	$(NVCC) -o $@ $< -c
query.o: query.c query.h
	$(CC) -o $@ $< -c
ir_collection_handler.o: ir_collection_handler.c ir_collection_handler.h ir_collection.h
	$(CC) -o $@ $< -c
ir_collection.o: ir_collection.c ir_collection.h
	$(CC) -o $@ $< -c

gpu_handler_simulation.o: gpu_handler_simulation.c gpu_handler.h query.h docscores.h ir_collection_handler.h
	$(CC) -o $@ $< -c


clean:
	rm -f *.o
