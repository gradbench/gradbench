EXECUTABLES=hello llsq lse


include ../../cpp/common.mk

LDFLAGS+=-lm
CXXFLAGS+=$(shell pkg-config --cflags eigen3) -fsanitize=address -g
