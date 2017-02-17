QT				+= widgets concurrent

TEMPLATE		 = app

CONFIG		+= c++11
QMAKESPEC = macx-g++
QMAKE_CXX = g++-6
QMAKE_CXXFLAGS += -std=c++11
QMAKE_LN = g++-6
QMAKE_LFLAGS = -fopenmp

LIBS			+= -L kernels/lib -lskepuimg -framework OpenCL
INCLUDEPATH	+= kernels/include ../../include

FORMS			 = mainwindow.ui
SOURCES		 = main.cpp lodepng.cpp
