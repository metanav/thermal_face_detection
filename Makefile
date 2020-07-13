PLATFORM?=RASPBERRYPI

CC = gcc
CFLAGS = -c -Wall -fpic  -Ilib/eve/include -I/usr/include/python3.7 
LD = $(CC)
LDFLAGS = -shared
PROGRAM = gd3
TARGET  = _$(PROGRAM).so
SWIG = swig 
SRC = gd3.c gd3_wrap.c
OBJECTS = $(patsubst %.c, %.o, $(shell find lib -name "*.c") $(SRC))
INTERFACE = $(PROGRAM)_wrap.c

all: $(INTERFACE) $(TARGET)

$(INTERFACE): 
	$(SWIG) -python gd3.i

$(TARGET):  $(OBJECTS)
	$(LD) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm $(TARGET) $(OBJECTS) gd3_wrap.c gd3.py

