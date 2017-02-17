################################################################################
#                     ____  _        ____  _   _   ____                        #
#                    / ___|| | _____|  _ \| | | | |___ \                       #
#                    \___ \| |/ / _ \ |_) | | | |   __) |                      #
#                     ___) |   <  __/  __/| |_| |  / __/                       #
#                    |____/|_|\_\___|_|    \___/  |_____|                      #
#                                                                              #
#                          ~ SkePU 2 main Makefile ~                           #
################################################################################


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

OUT_DIR = lib
TMP_DIR = tmp
SRC_DIR = src
TARGET_LIB = libskepuimg.a

SOURCES = invert edgedetect generate

PRECOMPILED_SOURCES = $(addsuffix .$(FILETYPE), $(addprefix $(TMP_DIR)/, $(SOURCES)))
OBJECTS = $(addsuffix .o, $(addprefix $(TMP_DIR)/, $(SOURCES)))

include Makefile.skepu


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

$(TMP_DIR)/edgedetect.$(FILETYPE): $(SRC_DIR)/edgedetect.cpp
	$(DBGR) $(SKEPU) -name edgedetect $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

$(TMP_DIR)/invert.$(FILETYPE): $(SRC_DIR)/invert.cpp
	$(DBGR) $(SKEPU) -name invert $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

$(TMP_DIR)/generate.$(FILETYPE): $(SRC_DIR)/generate.cpp
	$(DBGR) $(SKEPU) -name generate $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

%.o: %.cpp
	$(CXX) -c $(TARGET_FLAGS) -o $@ $<

# Precompiles and builds a single parallel test program.
lib: $(OBJECTS)
	# $(DBGR) $(CXX) $(OBJECTS) -o libskepuimg.a $(TARGET_FLAGS)
	ar rcs $(OUT_DIR)/$(TARGET_LIB) $(OBJECTS) 


# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	-$(RM) $(TMP_DIR)/*
	-$(RM) $(OUT_DIR)/$(TARGET_LIB)
