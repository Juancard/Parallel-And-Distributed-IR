CC=gcc
NVCC=gcc
OBJ=ir_collection.o ir_collection_handler.o gpu_handler.o docscores.o server_socket.o init.o
install: $(OBJ)
	$(NVCC) -o gpu_server $^
init.o: init.c ir_collection_handler.h server_socket.h
	$(NVCC) -o $@ $< -c
server_socket.o: server_socket.c docscores.h gpu_handler.h server_socket.h my_socket.h
	$(NVCC) -o $@ $< -c
docscores.o: docscores.c docscores.h
	$(NVCC) -o $@ $< -c
gpu_handler.o: gpu_handler.c gpu_handler.h docscores.h
	$(NVCC) -o $@ $< -c
ir_collection_handler.o: ir_collection_handler.c ir_collection_handler.h ir_collection.h
	$(NVCC) -o $@ $< -c
ir_collection.o: ir_collection.c ir_collection.h
	$(NVCC) -o $@ $< -c
clean:
	rm -f *.o
