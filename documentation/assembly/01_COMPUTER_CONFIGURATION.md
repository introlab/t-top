# Computer Configuration

## A. OpenCR Dev Rule
1. Copy [100-opencr-custom.rules](../../firmwares/opencr_firmware/100-opencr-custom.rules) in `/etc/udev/rules.d/`.
2. Add the user to the `dialout` group.
```bash
sudo usermod -a -G dialout $USER
```
