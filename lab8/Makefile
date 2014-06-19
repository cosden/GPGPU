#
#  Copyright 2012 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


CC       = pgcc
CCAP=35
CUDA=5.5
CCFLAGS  = $(FLAGS_FAST) 
ACCFLAGS = -Minfo=accel -acc -ta=nvidia,cc$(CCAP),$(CUDA)
BIN =  lab8-2
ENVVARS =

ifneq ($(dev), )
DEVID=$(dev)
endif

ifeq ($(time),1)
ENVVARS+=PGI_ACC_TIME=1
endif



default: release

release: lab8-2-solution.c
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $(BIN) $<

run:
	$(ENVVARS) ./$(BIN) $(DEVID)

clean:
	$(RM) $(BIN)
	$(RM) ./*.o
	$(RM) ./*.mod

help:
	@echo "USAGE: make [ release | run | clean | help ]"
	@echo
	@echo "OPTIONS:"
	@echo "    make run dev=<GPU device ID>   Runs the program explicitly on the device specified (ID>=0)."
	@echo "    make [release] time=1          Enables collecting simple timing information. Default: false."
	
