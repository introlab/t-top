#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`

cd $SCRIPT_PATH/..

npm run serve &
SERVER_PID=$!
trap "kill ${SERVER_PID}; exit 1" INT
sleep 10


export DISPLAY=:0.0
URL="http://localhost:8080/face"

if which xdg-open > /dev/null
then
  xdg-open $URL &> /dev/null
elif which gnome-open > /dev/null
then
  gnome-open $URL &> /dev/null
elif open > /dev/null
then
  open $URL &> /dev/null
else
  echo "The OS is not supported"
fi

wait $SERVER_PID
