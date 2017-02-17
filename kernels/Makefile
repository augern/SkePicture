##########################
# Makefile for SkePicture
##########################

include Makefile.skepu

# ---------------------------------------------------------------------------- #
# Test program names and target directories.

OUT_DIR = lib
TMP_DIR = tmp
SRC_DIR = src
TARGET_LIB = libskepuimg.a

SOURCES = invert edgedetect generate

PRECOMPILED_SOURCES = $(addsuffix .$(FILETYPE), $(addprefix $(TMP_DIR)/, $(SOURCES)))
OBJECTS = $(addsuffix .o, $(addprefix $(TMP_DIR)/, $(SOURCES)))


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

$(TMP_DIR)/edgedetect.$(BACK_EXT): $(SRC_DIR)/edgedetect.cpp
	$(DBGR) $(SKEPU) -name edgedetect $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

$(TMP_DIR)/invert.$(BACK_EXT): $(SRC_DIR)/invert.cpp
	$(DBGR) $(SKEPU) -name invert $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

$(TMP_DIR)/generate.$(BACK_EXT): $(SRC_DIR)/generate.cpp
	$(DBGR) $(SKEPU) -name generate $<  -dir $(TMP_DIR) $(SKEPU_FLAGS)

%.o: %.$(BACK_EXT)
	$(BACK_CXX) -c $(TARGET_FLAGS) -o $@ $<

# Precompiles and builds a single parallel test program.
lib: $(OBJECTS)
	ar rcs $(OUT_DIR)/$(TARGET_LIB) $(OBJECTS) 

# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	-$(RM) $(TMP_DIR)/*
	-$(RM) $(OUT_DIR)/$(TARGET_LIB)
