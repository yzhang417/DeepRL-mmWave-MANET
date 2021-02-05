SHELL := /bin/bash

cuda0:
	script -f -c 'python mainDRL.py --iterations 100 --batch 40 --slots 2000' output/mylog.log
cuda1:
	script -f -c 'python mainDRL.py --iterations 100 --batch 40 --slots 2000' output/mylog.log
