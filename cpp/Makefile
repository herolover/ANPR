.PHONY: clean

program_name := ./bin/ANPR
source_dirs  := .
include_dirs := .
link_flags   := -lopencv_highgui -lopencv_core -lopencv_imgproc -lrt -lboost_system -lboost_filesystem -lopencv_photo -ltesseract
lib_dirs     := .
compiler_flags :=  -Wall -g -std=c++11

output_path := ./out/

#source_dirs  := $(addprefix ./src/, $(source_dirs))
source_files := $(wildcard $(addsuffix /*.cpp, $(source_dirs)))
object_files := $(notdir $(source_files))
object_files := $(object_files:.cpp=.o)

object_files2 := $(addprefix $(output_path), $(object_files))

VPATH := $(source_dirs) $(output_path)

include_flags := $(addprefix -I, $(include_dirs)) $(addprefix -I, $(source_dirs))
lib_flags := $(addprefix -L, $(lib_dirs))

$(program_name): $(object_files)
	g++ $(compiler_flags) $(object_files2) $(include_flags) $(lib_flags) -o $@ $(link_flags) -pipe

%.o: %.cpp
	#astyle --style=gnu -t $<
	#rm -f *.orig
	g++ $(compiler_flags) -c -MD $(include_flags) -o $(output_path)$@ $<

clean:
	rm $(output_path)* *.d *.o *~ \#*\# *.out

include $(wildcard *.d)
