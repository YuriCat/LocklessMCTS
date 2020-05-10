CXX      = c++
CXXFLAGS = -std=c++11 -MMD -MP
OPT      = -O3 -march=native -DNDEBUG 
#OPT     := -O0 -g -ggdb -D_GLIBCXX_DEBUG
LDFLAGS  = -pthread
LIBS     =
INCLUDES =
SRC_DIR  = ./
OBJ_DIR  = ./obj
SRCS     = $(wildcard $(SRC_DIR)/*.cpp)
OBJS     = $(subst $(SRC_DIR),$(OBJ_DIR), $(SRCS:.cpp=.o))
TARGET   = main
DEPENDS  = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS) $(LIBS)
	$(CXX) $(OPT) -o $@ $(OBJS) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@if [ ! -d $(OBJ_DIR) ]; \
		then echo "mkdir -p $(OBJ_DIR)"; mkdir -p $(OBJ_DIR); \
		fi
	$(CXX) $(CXXFLAGS) $(OPT) $(INCLUDES) -o $@ -c $< 

clean:
	$(RM) -r $(OBJ_DIR) $(TARGET)

-include $(DEPENDS)

.PHONY: all clean
