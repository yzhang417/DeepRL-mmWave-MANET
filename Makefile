SHELL := /bin/bash
batch20:
	script -f -c 'python mainDRL.py --Netw_topo_id 4 --iterations 500 --batch 20 --slots 2000 --output batch20' logfiles/batch20.log
batch60:
	script -f -c 'python mainDRL.py --Netw_topo_id 3 --iterations 500 --batch 60 --slots 2000 --output batch60' logfiles/batch60.log
batch100:
	script -f -c 'python mainDRL.py --Netw_topo_id 3 --iterations 500 --batch 100 --slots 2000 --output batch100' logfiles/batch100.log
