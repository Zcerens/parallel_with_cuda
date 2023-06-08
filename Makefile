CC = nvcc
CFLAGS = `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`
TARGET = image_resize

all: $(TARGET)

$(TARGET): image_resize.cu
	$(CC) -o $(TARGET) image_resize.cu $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
