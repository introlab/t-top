#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <robot_ip>"
    exit 1
fi

# Get the robot IP from the command line
IP=$1

# Move to directory
mkdir -p ~/.ros/opentera/certs
pushd ~/.ros/opentera/certs > /dev/null

# Remove old certificate
rm ca-cert.pem ca-key.pem server-req.pem server-key.pem server-cert.pem

# 1. Generate CA's private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 -days 365 -keyout ca-key.pem -out ca-cert.pem -subj "/C=CA/ST=Quebec/L=Sherbrooke/O=IntRoLab" -passout "pass:introlab"

# 2. Create a certificate request for the server
# Make sure that the IP address match the robot's IP address
openssl req -newkey rsa:4096 -keyout server-key.pem -out server-req.pem -subj "/C=CA/ST=Quebec/L=Sherbrooke/O=IntRoLab/CN=$IP" -nodes

# 3. Sign server request using CA's private key and certificate
openssl x509 -req -in server-req.pem -days 365 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -passin "pass:introlab"

echo "Server certificate validity test:"
openssl verify -CAfile ca-cert.pem server-cert.pem

# You can now copy the ca-cert.pem to the target machine
# Then, rename the ca-cert.pem file to <robot>-ca-cert.crt
# Copy it to /usr/local/share/ca-sertificates
# Update the certificates using 'sudo update-ca-certificates --fresh'

popd > /dev/null
