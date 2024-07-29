#!/bin/csh

@ i = 0
while ( {qstat -u aponte | grep -q 9929314} )
    echo "Job not finished $i" > tmp.log
    sleep 2
    @ i += 1
end

