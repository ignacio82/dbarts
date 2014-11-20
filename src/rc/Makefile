INCLUDE_DIR=../include

PKG_CPPFLAGS=-I$(INCLUDE_DIR)
ALL_CPPFLAGS=$(R_XTRA_CPPFLAGS) $(PKG_CPPFLAGS) $(CPPFLAGS)

LOCAL_SOURCES=rc.c
LOCAL_OBJECTS=rc.o

all : librc.a

librc.a : $(LOCAL_OBJECTS)
	$(AR) $(ARFLAGS) librc.a $(LOCAL_OBJECTS)

clean :
	-rm -f *.o
	-rm -f *.a

rebuild : clean all

$(INCLUDE_DIR)/rc/rc.h :

rc.o : rc.c $(INCLUDE_DIR)/rc/rc.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c rc.c -o rc.o
