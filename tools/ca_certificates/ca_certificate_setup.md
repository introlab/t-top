# Generating CA Certificates for the robot

## 1. Generate CA Certificates on the robot
You need to generate a CA certificate for the robot to allow for HTTPS connections to the robot on the local network.
It will be tied to its IP address on the LAN, so make sure to give the robot a static IP address before generating the certificate.
Make sure to replace the `<IP_ADDRESS_OF_THE_ROBOT>` placeholder with the IP address of the robot in the command below.

```bash
# Run this script on the robot
./generate_certificate.sh <IP_ADDRESS_OF_THE_ROBOT>
```

This will need to be done again when the certificate expires, after 365 days.

## 2. Copy the CA certificate to the developpement machine
If the developpement machine is using Ubuntu, you can use a script to install the certificate. The script uses `scp` to copy the certificate to the developpement machine, so you will need the password for the `introlab` user on the robot if you don't have an ssh key setup.
Make sure to replace the `<IP_ADDRESS_OF_THE_ROBOT>` placeholder with the IP address of the robot (for SSH) and the `<NAME_OF_THE_ROBOT>` placeholder with the name of the robot (for the certificate file name, to distinguish it from files from other robots).

```bash
# Run this script on the developpement machine
./fetch_certificate.sh <IP_ADDRESS_OF_THE_ROBOT> <NAME_OF_THE_ROBOT>
```

This step can be done by many developpers without the need to redo step one every time, but it will have to be redone if step 1 is done again.

## 3. Add the CA certificate to your browser
The certificate installed on your system is not enough for your browser to trust it.
You need to add the certificate to your browser.

Open your browser's settings, and find the CA Certificates setting. Add the ca-certificate `/usr/local/share/ca-sertificates/<NAME_OF_THE_ROBOT>-ca-cert.crt` as a trusted certificate.

### Firefox
1. Open the Preferences menu
2. Search for "certif"
3. Click on the "Show Certificates" button
4. Go to the "Authorities" tab
5. Click on the "Import" button
6. Select the certificate file

### Chromium (Chrome, Brave, etc.)
1. Navigate to the URL  ̀chrome://settings/certificates`
2. Go to the "Authorities" tab
3. Click on the "Import" button
4. Select the certificate file
