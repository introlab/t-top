#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <robot_ip> <robot_distinct_name>"
    exit 1
fi

# Get the robot IP and name from the command line
IP=$1
NAME=$2

# Fetch CA certificate from the robot
scp introlab@$IP:/home/introlab/.ros/opentera/certs/ca-cert.pem /tmp/ca-cert.pem

# Move to installation folder
sudo mv /tmp/ca-cert.pem /usr/local/share/ca-certificates/$NAME-ca-cert.crt

# Install certificate
sudo update-ca-certificates --fresh
