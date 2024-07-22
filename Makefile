CC = nvcc
CFLAGS = -I. -g

DEPS = conv3d.h autoencoder.h loss.h utils.h
OBJ = main.o conv3d.o autoencoder.o loss.o utils.o

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

autoencoder: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o autoencoder
