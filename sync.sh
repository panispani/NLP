#!/bin/bash

while true; do
    rsync -rav -e "sshpass -p 'root' ssh -p$nport" root@0.tcp.ngrok.io:~/model.bin ./
    sleep 60
done
