#!/bin/bash

SCRIPT=`realpath --no-symlinks $0`
SCRIPT_PATH=`dirname $SCRIPT`

cd $SCRIPT_PATH/..

chmod +x $SCRIPT_PATH/node_modules/serve/build/main.js
$SCRIPT_PATH/node_modules/serve/build/main.js -S -s $SCRIPT_PATH/dist -l 8080 &
SERVER_PID=$!
trap "kill ${SERVER_PID}; exit 1" INT
sleep 1

if [ "$1" == "true" ]
then
  if [ "$2" == "true" ]
  then
    FULLSCREEN="--kiosk"
  else
    FULLSCREEN=""
  fi

  export DISPLAY=:0.0
  URL="http://localhost:8080/face"

  if which chromium-browser > /dev/null
  then
    chromium-browser $FULLSCREEN $URL &> /dev/null
  elif which xdg-open > /dev/null
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
fi

wait $SERVER_PID
