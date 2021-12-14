#!/bin/bash

SCRIPT=`realpath $0`
SCRIPT_PATH=`dirname $SCRIPT`

cd $SCRIPT_PATH/..
npm install

npm run serve &
SERVER_PID=$!
trap "kill ${SERVER_PID}; exit 1" INT
sleep 10


URL="http://localhost:8080/face"

if which xdg-open > /dev/null
then
  xdg-open $URL
elif which gnome-open > /dev/null
then
  gnome-open $URL
elif open > /dev/null
then
  open $URL
else
  echo "The OS is not supported"
fi

wait $SERVER_PID
