CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(week_01)

FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
ADD_EXECUTABLE( week_01 main.cc )
TARGET_LINK_LIBRARIES( week_01 ${OpenCV_LIBS} )